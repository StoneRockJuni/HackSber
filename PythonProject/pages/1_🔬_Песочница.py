"""«Песочница»: загрузить ЛЮБОЕ фото и проверить, что система о нём думает.

Страница нужна для двух вещей:

1. Ответить на вопрос «откуда вы берёте GPS?» - показать источник координат,
   сырые теги, независимые проверки и честное «координат нет».
2. Проверить нейросеть на любом снимке (в том числе найденном в интернете):
   увидеть рамки, стадию, уверенность и решение «авто / к инспектору».

Это отдельная страница Streamlit, основной сценарий банка живёт в app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from net_safe import fetch_image  # noqa: E402
from photo_ai import analyze_frame, fuse_frames, render_overlay, to_display_rgb  # noqa: E402
from sber_vision import CLASS_NAMES_RU, model_available, model_sha256  # noqa: E402
from security import evidence_level, extract_exif_data, gps_provenance, run_quality_gate  # noqa: E402

st.set_page_config(page_title="Песочница проверки фото", layout="wide", page_icon="🔬")

SOURCE_LABELS = {
    "exif": "EXIF GPS IFD (метаданные самого файла)",
    "xmp": "XMP-пакет внутри файла",
    "flight_log": "Имя файла (выгрузка полётного лога)",
    "none": "Координат нет",
}


def _verdict(fused: dict, gate: dict, prov: dict, evidence: dict | None = None) -> tuple[str, str]:
    evidence = evidence or {}
    if not gate.get("image_quality_passed"):
        return "error", "Снимок не прошёл контроль качества: нужен другой кадр."
    if evidence.get("level") == "none":
        return (
            "error",
            "Снимок не к чему привязать: нет ни GPS, ни QR-маркера полётного задания. "
            "Это риск фрода — автоматика блокируется.",
        )
    if evidence.get("level") == "weak":
        return (
            "warning",
            f"Снимок доказан частично: {evidence.get('reason')}. Это не фрод, но и не "
            "основание для авторешения — нужен повторный кадр с маркером или инспектор.",
        )
    if prov["used"]["source"] == "none":
        return (
            "warning",
            "Геопривязки нет. Для банка это «не подтверждено», а не «объект не построен». "
            "Фото из соцсетей и мессенджеров почти всегда без GPS — это свойство файла.",
        )
    if fused["needs_manual_review"]:
        return (
            "warning",
            f"Автоматическое решение не принимается (уверенность {fused['confidence']:.2f} < 0.75): "
            "повторный облёт или ручной разбор инспектора.",
        )
    if fused["stage"] == "100%":
        return "success", "Закрытый контур: можно фиксировать 100% и переходить к оформлению/залогу."
    if fused["stage"] == "50%":
        return "info", "Контур не закрыт: фундамент/стены без готовой кровли — стадия 50%."
    return "info", "Элементов дома не найдено — похоже на 0%, требуется подтверждение."


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _show_gps(prov: dict, exif: dict) -> None:
    used = prov["used"]
    st.subheader("1. Откуда взялись координаты")
    if used["source"] == "none":
        st.error("GPS: НЕ НАЙДЕН — ни EXIF, ни XMP, ни имя файла.")
    else:
        altitude = f", высота {used['alt_m']} м" if used.get("alt_m") is not None else ""
        st.success(
            f"GPS: **{used['lat']}, {used['lon']}**{altitude}\n\n"
            f"Источник: **{SOURCE_LABELS.get(used['source'], used['source'])}**"
        )
    st.caption(prov["explanation"])

    if prov["sources"]:
        st.markdown("**Все найденные источники координат:**")
        st.dataframe(
            [
                {
                    "источник": SOURCE_LABELS.get(item["source"], item["source"]),
                    "широта": item["lat"],
                    "долгота": item["lon"],
                    "высота, м": item.get("alt_m"),
                    "доверенный": "да" if item["trusted"] else "нет",
                }
                for item in prov["sources"]
            ],
            width="stretch",
            hide_index=True,
        )
    if prov["checks"]:
        st.markdown("**Независимые проверки метаданных:**")
        st.dataframe(
            [
                {"проверка": check["name"], "ок": "✅" if check["ok"] else "❌", "детали": check["detail"]}
                for check in prov["checks"]
            ],
            width="stretch",
            hide_index=True,
        )
    with st.expander("Сырые метаданные (диагностика)"):
        st.json(
            {
                "captured_at": exif.get("captured_at"),
                "camera_make": exif.get("camera_make"),
                "drone_model": exif.get("drone_model"),
                "gps_source": exif.get("gps_source"),
                "xmp": prov.get("xmp_metadata"),
            }
        )


def _show_gate(gate: dict, gate_demo: dict | None, has_cadastral: bool) -> None:
    meta = gate.get("_meta", {})
    st.subheader("2. Quality Gate")
    cols = st.columns(4)
    cols[0].metric("Резкость (Laplacian)", f"{meta.get('sharpness')}", help="норма > 120")
    cols[1].metric("Яркость", f"{meta.get('brightness')}")
    cols[2].metric("GPS совпадает", "да" if gate.get("exif_gps_match") else "нет")
    cols[3].metric("QR/маркер", "есть" if gate.get("challenge_marker_present") else "нет")
    if gate.get("error_message"):
        st.warning(gate["error_message"])
    st.caption(
        "Режим проверки GPS: "
        + ("точная кадастровая точка" if has_cadastral else "региональный fallback (Томская область)")
        + ". Строгий режим требует QR-маркер полётного задания; демо-режим считает свежестью любой EXIF."
    )


def _show_ai(frame: dict | None, fused: dict | None, frames: list[dict], title: str) -> dict | None:
    st.subheader("3. Нейросеть (YOLOv8n SberStroi, ONNX, локально)")
    if not model_available():
        st.info("Модель не найдена: models/sberstroi_yolov8n.onnx. Доступен только OpenCV fallback.")
        return None
    if not frame or not fused:
        st.info("Кадр не удалось декодировать.")
        return None

    cols = st.columns(4)
    cols[0].metric("Стадия", fused["stage"])
    cols[1].metric("Уверенность", f"{fused['confidence']:.2f}")
    cols[2].metric("Кадров в серии", fused.get("frames", len(frames)))
    cols[3].metric("Согласованность", f"{fused.get('agreement', 0):.0%}")

    st.caption(
        f"Причина вердикта: `{fused['reason']}` · "
        f"голоса: 100% ×{fused.get('evidence', {}).get('frames_100', 0)}, "
        f"50% ×{fused.get('evidence', {}).get('frames_50', 0)}, "
        f"0% ×{fused.get('evidence', {}).get('frames_0', 0)}"
    )
    classes = frame.get("classes") or []
    st.write(
        "**Найденные элементы на кадре:** "
        + (", ".join(CLASS_NAMES_RU.get(c, c) for c in classes) if classes else "нет")
    )
    st.caption(
        "Белый прямоугольник — зона «объект интереса»: детекции в центре кадра считаются целевым "
        "домом, детекции у краёв (соседские кровли, заборы) не используются в решении."
    )

    overlay = render_overlay(title, frame, stage=fused["stage"], confidence=fused["confidence"])
    if overlay is not None:
        st.image(_bgr_to_rgb(overlay), caption="Разметка: рамки детектора и зона объекта", width="stretch")

    if len(frames) > 1:
        st.markdown("**По кадрам:**")
        st.dataframe(
            [
                {
                    "кадр": index,
                    "стадия": item["stage"],
                    "уверенность": round(item["confidence"], 3),
                    "причина": item["reason"],
                    "элементы": ", ".join(item["classes"]) or "-",
                }
                for index, item in enumerate(frames, start=1)
            ],
            width="stretch",
            hide_index=True,
        )
    return fused


st.title("🔬 Песочница: проверить любое фото")
st.caption(
    "Загрузите снимок с телефона, из интернета или серию кадров дрона. "
    "Система покажет, откуда взяты координаты, что видит нейросеть и какое решение приняла бы."
)

with st.sidebar:
    st.header("Параметры проверки")
    marker = st.text_input("Ожидаемый QR-маркер", value="", help="Пусто = принимать любой QR")
    st.markdown("**Кадастровая точка (необязательно)**")
    col_lat, col_lon = st.columns(2)
    lat = col_lat.number_input("Широта", value=0.0, format="%.6f", step=0.0001)
    lon = col_lon.number_input("Долгота", value=0.0, format="%.6f", step=0.0001)
    use_point = st.checkbox("Сверять GPS с точкой", value=False)
    st.divider()
    st.caption(
        "Фото никуда не отправляется: ссылка скачивается локально с фильтром внутренних адресов, "
        "нейросеть работает на этой машине."
    )
    if model_available():
        st.success("Модель ONNX подключена")
        st.caption(f"SHA-256: `{str(model_sha256())[:24]}…`")
    else:
        st.warning("Модель ONNX не найдена")

tab_file, tab_url = st.tabs(["📁 Файлы с компьютера", "🌐 Ссылка из интернета"])

raw_items: list[tuple[str, bytes]] = []

with tab_file:
    uploaded = st.file_uploader(
        "Один снимок или серия кадров одного объекта",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="sandbox_files",
    )
    if uploaded:
        raw_items = [(item.name, item.getvalue()) for item in uploaded]

with tab_url:
    url = st.text_input("Прямая ссылка на изображение (http/https)", placeholder="https://example.com/house.jpg")
    if st.button("Скачать и проверить", type="secondary"):
        if url:
            try:
                fetched = fetch_image(url)
                name = Path(url.split("?")[0]).name or "downloaded.jpg"
                st.session_state["url_image"] = (name, fetched.data)
                st.success(f"Скачано {len(fetched.data) / 1024:.0f} КБ ({fetched.content_type})")
            except Exception as exc:  # noqa: BLE001 - показываем причину отказа человеку
                st.session_state.pop("url_image", None)
                st.error(f"Не удалось скачать: {exc}")
        else:
            st.warning("Вставьте ссылку")
    if st.session_state.get("url_image"):
        name, data = st.session_state["url_image"]
        st.caption(f"В памяти: `{name}` ({len(data) / 1024:.0f} КБ)")
        raw_items = [(name, data)]

if not raw_items:
    st.info(
        "Загрузите фото слева/сверху. Хороший тест: снимок дома из интернета (координат не будет — "
        "это ожидаемо), скриншот, картинка-коллаж, реальный кадр дрона с EXIF."
    )
    st.stop()

first_name, first_bytes = raw_items[0]
cadastral = (lat, lon) if use_point else None
gate_args = {
    "filename": first_name,
    "expected_marker": marker or None,
    "cadastral_gps": cadastral,
    "tolerance_m": 30.0,
}

try:
    gate_strict = run_quality_gate(first_bytes, allow_exif_fallback=False, **gate_args)
    gate_demo = run_quality_gate(first_bytes, allow_exif_fallback=True, **gate_args)
    prov = gps_provenance(first_bytes, first_name)
    exif = extract_exif_data(first_bytes)
    evidence = evidence_level(gate_strict, gate_strict.get("_meta"))
except Exception as exc:  # noqa: BLE001
    st.error(f"Не удалось разобрать файл: {exc}")
    st.stop()

frames: list[dict] = []
for name, data in raw_items[:24]:
    report = analyze_frame(data)
    if report:
        frames.append(report)
fused = fuse_frames(frames) if frames else None

best_index = 0
if frames and fused:
    best_index = max(
        range(len(frames)),
        key=lambda i: (frames[i]["stage"] == fused["stage"], frames[i]["confidence"]),
    )
best_name, best_bytes = raw_items[best_index]

level, message = _verdict(
    fused or {"stage": "0%", "confidence": 0.0, "needs_manual_review": True}, gate_strict, prov, evidence
)
{"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[level](message)

EVIDENCE_LABELS = {
    "strong": "сильная (GPS + QR-маркер)",
    "medium": "средняя (GPS + EXIF, демо-режим)",
    "weak": "частичная (что-то одно)",
    "none": "нет доказательств",
}
st.caption(
    f"**Доказанность снимка:** {EVIDENCE_LABELS.get(evidence['level'], evidence['level'])} — "
    f"{evidence['reason']}. {evidence['note']}"
)

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Загруженный кадр")
    try:
        preview = to_display_rgb(best_bytes)
        if preview is not None:
            st.image(preview, caption=f"{best_name} ({len(best_bytes) / 1024:.0f} КБ)", width="stretch")
        else:
            st.warning(
                "Предпросмотр недоступен: файл не декодируется как изображение. "
                "На анализ это не влияет — смотрите блоки ниже."
            )
    except Exception as exc:  # noqa: BLE001 - превью не должно ронять страницу
        st.warning(f"Предпросмотр не отрисован ({exc}). На анализ это не влияет.")
with right:
    _show_gps(prov, exif)
    _show_gate(gate_strict, gate_demo, bool(cadastral))

st.divider()
_show_ai(frames[best_index] if frames else None, fused, frames, best_bytes)

st.divider()
st.caption(
    "⚠️ Фото из интернета — out-of-distribution для этой модели: другой ракурс, камера, сжатие, "
    "иногда рендер или коллаж. Такой тест проверяет устойчивость, но НЕ измеряет точность. "
    "Точность измеряется только на размеченном наборе кадров с БПЛА "
    "(см. tools/eval_model.py и docs/AI_VALIDATION.md)."
)
