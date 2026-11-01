import streamlit as st
import pandas as pd
import datetime
import json
import os
from vision_agent import analyze_object_images
from director import evaluate_strategy
from photo_ai import to_display_rgb
from sber_vision import model_available
from security import (
    audit_chain_report,
    build_snapshot_passport,
    compute_image_hash,
    verify_audit_chain,
    write_audit_log,
)

AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit_trail.jsonl")

st.set_page_config(layout="wide", page_title="СтройКонтур | Сибирский Банк", page_icon="🚁")

# Стилизация метрик
st.markdown("""
<style>
.big-badge { font-size: 18px; font-weight: bold; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🚁 АС «СтройКонтур» — AI-мониторинг объектов ИЖС")
st.caption(
    "Сибирский Банк | Автоматическая экспертиза аэрофотоснимков с БПЛА («Глаз») + Принятие решений («AI-Дирижёр»)")


# Загрузка реального реестра
@st.cache_data
def load_registry():
    return pd.read_excel("Реестр_для_теста.xlsx")


try:
    df_reg = load_registry()
except Exception as e:
    st.error(f"Положите файл 'Реестр_для_теста.xlsx' в папку проекта! Ошибка: {e}")
    st.stop()

# Боковая панель управления
st.sidebar.header("🎛️ Пульт инспектора ИЖС")
selected_idx = st.sidebar.selectbox(
    "Выберите объект из Реестра (34 участка):",
    df_reg.index,
    format_func=lambda
        i: f"#{i + 1} | {df_reg.loc[i, 'Кадастровый номер']} ({str(df_reg.loc[i, 'Территория']).split(',')[-1].strip()})"
)
row = df_reg.loc[selected_idx].to_dict()

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Карточка кредитного досье")
st.sidebar.write(f"**Адрес:** {row['Территория']}")
st.sidebar.write(f"**Кадастровый №:** `{row['Кадастровый номер']}`")
st.sidebar.write(f"**Старт стройки:** `{row['Дата начала строительства']}`")
st.sidebar.write(f"**Плановая дата:** `{row['Плановая дата']}`")
st.sidebar.write(f"**Наличие ПЗ (просрочки):** `{row['Наличие ПЗ']}`")
st.sidebar.write(f"**ОН зарегистрирован:** `{row['ОН зарегистрирован']}` | **В залоге:** `{row['Залог']}`")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Параметры симуляции")
contact_ok = st.sidebar.checkbox("Клиент выходит на связь (Успешная коммуникация)", value=True)
override_pz = st.sidebar.selectbox("Тест сценария ПЗ (Просрочки):",
                                   ["Из реестра (" + str(row['Наличие ПЗ']) + ")", "Да (Тест просрочки)",
                                    "Нет (Без просрочки)"])
if "Да (" in override_pz:
    row["Наличие ПЗ"] = "Да"
elif "Нет (" in override_pz:
    row["Наличие ПЗ"] = "Нет"

api_key = st.sidebar.text_input("Gemini API Key (только демо, не для реальных ПДн):", type="password",
                                help="Внешний VLM выключен по умолчанию. Для включения задайте переменную окружения ALLOW_EXTERNAL_VLM=1. Реальные снимки и ПДн отправлять нельзя.")
expected_marker = st.sidebar.text_input(
    "QR-маркер полётного задания (demo):",
    value="STROYKONTUR-DEMO-2026",
    help="Одноразовый маркер, который подрядчик обязан снять на объекте. Оставьте поле пустым, чтобы принимать любой QR.",
)

if model_available():
    st.sidebar.success("AI: YOLOv8n SberStroi (ONNX) подключён")
else:
    st.sidebar.info("AI: OpenCV fallback (ONNX-модель не найдена)")

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Безопасность и аудит")
audit_placeholder = st.sidebar.empty()


def render_audit_status():
    report = audit_chain_report(AUDIT_LOG_PATH)
    if report["valid"]:
        audit_placeholder.success(f"Audit Trail: цепочка цела ({report['records']} записей)")
    else:
        audit_placeholder.error(f"Audit Trail: нарушение целостности — {report['error']}")
    return report


render_audit_status()
if st.sidebar.button("Проверить целостность журнала", width="stretch"):
    if verify_audit_chain(AUDIT_LOG_PATH):
        st.sidebar.info("Проверка пройдена: все hash_prev и record_hash совпадают.")
    else:
        st.sidebar.error("Проверка не пройдена: журнал изменён или повреждён.")
st.sidebar.caption(f"Журнал: `{AUDIT_LOG_PATH}` · SHA-256 hash chain · append-only")

# Верхняя панель KPI банка
k1, k2, k3, k4 = st.columns(4)
k1.metric("Портфель ИЖС Сибирского банка", "17 800 кредитов", "Контроль 3–5 раз за цикл")
k2.metric("Скорость осмотра 20 объектов", "1 день (БПЛА)", "-13 дней (ускорение в 14 раз)")
k3.metric("Выбранный кадастровый участок", row["Кадастровый номер"])
k4.metric("Плановый срок завершения", str(row["Плановая дата"]), f"ПЗ: {row['Наличие ПЗ']}")
st.caption(
    f"Полностью (без обрезки в плитках): участок `{row['Кадастровый номер']}` · "
    f"плановая дата `{row['Плановая дата']}` · ПЗ (просроченная задолженность): "
    f"`{row['Наличие ПЗ']}` · ОН зарегистрирован: `{row['ОН зарегистрирован']}` · "
    f"Залог: `{row['Залог']}`"
)

st.markdown("---")
col_left, col_right = st.columns([1.15, 1])

with col_left:
    st.subheader("👁️ 1. AI-Агент «Глаз» (Компьютерное зрение и Геодезия)")
    uploaded_files = st.file_uploader(
        "Загрузите аэрофотоснимки объекта с дрона (поддерживается пакетная загрузка всех ракурсов):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    passport = None
    if uploaded_files:
        # Отображение превью загруженных снимков
        img_cols = st.columns(min(len(uploaded_files), 4))
        for idx, file_obj in enumerate(uploaded_files[:4]):
            try:
                file_obj.seek(0)
                raw_preview = file_obj.read()
                file_obj.seek(0)
                preview = to_display_rgb(raw_preview)
                if preview is None:
                    raise ValueError("файл не декодируется как изображение")
                img_cols[idx].image(preview, caption=file_obj.name, width="stretch")
            except Exception:
                img_cols[idx].warning(f"Пропуск превью: `{file_obj.name}` не является изображением")
            finally:
                file_obj.seek(0)

        # Паспорт снимка: SHA-256 файла + EXIF + псевдоним клиента + полётное задание
        try:
            passport = build_snapshot_passport(
                uploaded_files[0],
                client_id=f"CL-{selected_idx + 1:04d}",
                drone_id="DJI-M3M-DEMO",
                flight_task_id=f"FT-{datetime.date.today().isoformat()}-{selected_idx + 1:03d}",
                operator_id="contractor-demo",
            )
        except Exception as passport_error:
            st.warning(f"Паспорт снимка не сформирован: {passport_error}")

        # Автоматический запуск анализа
        context_text = f"Кадастровый номер: {row['Кадастровый номер']}, Адрес: {row['Территория']}"
        eye_res = analyze_object_images(
            uploaded_files,
            cadastral_context=context_text,
            api_key=api_key,
            expected_marker=expected_marker,
        )
        meta = eye_res.get("_meta", {})
        gps = meta.get("gps", {})

        # Блок метаданных и Quality Gate
        m1, m2, m3, m4 = st.columns(4)
        m1.info(f"**Комплектация:**\n{gps.get('type', 'Тип А')}")
        m2.info(f"**GPS Снимка:**\n`{gps.get('lat')}, {gps.get('lon')}`")
        m3.info(f"**Резкость (Laplacian):**\n`{meta.get('sharpness')}` (Норма >120)")
        m4.info(f"**SHA-256:**\n`{meta.get('sha256_short', '—')}`")
        if meta.get("sha256"):
            st.caption(f"Полный SHA-256 снимка: `{meta['sha256']}`")
        if meta.get("expected_marker"):
            marker_ok = eye_res["quality_gate"].get("challenge_marker_present")
            st.caption(
                f"QR-маркер `{meta.get('expected_marker')}`: "
                f"{'подтверждён' if marker_ok else 'не найден'} "
                f"(источник: {meta.get('marker_source', 'none')})"
            )

        ai_meta = meta.get("ai") or {}
        if str(ai_meta.get("source", "")).startswith("sber_yolo_onnx"):
            detected = eye_res["construction_analysis"].get("detected_classes") or []
            st.caption(
                f"AI: YOLOv8n SberStroi · классов проанализировано: {ai_meta.get('frames_analyzed', 1)} · "
                f"классы: {', '.join(detected) if detected else 'нет'} · "
                f"уверенность стадии: {eye_res['construction_analysis'].get('confidence_stage')} · "
                f"согласованность серии: {ai_meta.get('agreement')} · "
                f"модель SHA-256: `{str(ai_meta.get('model_sha256'))[:16]}…`"
            )

        with st.expander("🔎 Диагностика: откуда GPS и почему такой вердикт", expanded=False):
            st.json(
                {
                    "gps_source": gps.get("source"),
                    "has_exif": gps.get("has_exif"),
                    "gps": {"lat": gps.get("lat"), "lon": gps.get("lon")},
                    "exif_date": gps.get("date"),
                    "drone_model": gps.get("drone_model"),
                    "sha256": meta.get("sha256"),
                    "sharpness": meta.get("sharpness"),
                    "brightness": meta.get("brightness"),
                    "marker_source": meta.get("marker_source"),
                    "marker_payload": meta.get("marker_payload"),
                    "quality_gate": eye_res.get("quality_gate"),
                    "ai": meta.get("ai"),
                }
            )
            st.caption(
                "GPS читается только из EXIF самого файла (Pillow, GPS IFD) или из имени файла "
                "вида `54.8511_83.1946.jpg`. Приложение не определяет геолокацию по IP, картам "
                "или устройству. Соцсети и мессенджеры часто удаляют EXIF — тогда источник будет `none`."
            )

        # Интерактивная корректировка (Human-in-the-loop) при необходимости протестировать крайние случаи на защите
        with st.expander("🛠️ Ручная верификация экспертом (Human-in-the-Loop / Тест сценариев ИБ)", expanded=False):
            c1, c2 = st.columns(2)
            stage_list = ["0%", "50%", "100%"]
            cur_stage_idx = stage_list.index(eye_res["construction_analysis"]["detected_stage"])
            eye_res["construction_analysis"]["detected_stage"] = c1.selectbox("Стадия готовности:", stage_list,
                                                                              index=cur_stage_idx)

            mat_list = ["gas_concrete", "brick", "wood", "monolith", "unknown"]
            cur_mat_idx = mat_list.index(eye_res["construction_analysis"]["wall_material"])
            eye_res["construction_analysis"]["wall_material"] = c2.selectbox("Материал стен:", mat_list,
                                                                             index=cur_mat_idx)

            eye_res["geospatial_check"]["boundaries_violation"] = c1.checkbox("Нарушение кадастровых границ (<3м)",
                                                                              value=eye_res["geospatial_check"][
                                                                                  "boundaries_violation"])
            eye_res["quality_gate"]["exif_gps_match"] = c2.checkbox("GPS совпадает с кадастром",
                                                                    value=eye_res["quality_gate"]["exif_gps_match"])
            eye_res["quality_gate"]["challenge_marker_present"] = c1.checkbox("Подлинность/маркер подтверждены",
                                                                              value=eye_res["quality_gate"][
                                                                                  "challenge_marker_present"])
            eye_res["quality_gate"]["image_quality_passed"] = c2.checkbox("Quality Gate пройден",
                                                                          value=eye_res["quality_gate"][
                                                                              "image_quality_passed"])

        # Отображение строгого JSON по ТЗ
        st.markdown("#### 📄 Выходной JSON-контракт AI-Агента «Глаз»:")
        clean_eye_json = {k: v for k, v in eye_res.items() if not k.startswith("_")}
        st.json(clean_eye_json)

        if passport:
            with st.expander("🛡️ Паспорт снимка / Chain of Custody", expanded=False):
                st.caption(
                    "Паспорт связывает файл, объект, полётное задание и оператора. "
                    "Хэш покрывает все байты файла, включая EXIF."
                )
                st.json(passport)
                st.caption(f"Проверка паспорта: {passport['verification_url']}")
                st.download_button(
                    "Скачать паспорт JSON",
                    data=json.dumps(passport, ensure_ascii=False, indent=2),
                    file_name=f"passport_{passport['passport_id'][:8]}.json",
                    mime="application/json",
                )

        # Отображение точки на карте, если есть координаты
        if gps.get("lat") and gps.get("lon"):
            with st.expander("🗺️ Посадка объекта на спутниковой/кадастровой карте по GPS из EXIF"):
                map_df = pd.DataFrame({"lat": [gps["lat"]], "lon": [gps["lon"]]})
                st.map(map_df, zoom=15)

with col_right:
    st.subheader("🧠 2. Слой «AI-Дирижёр» (Маршрутизация по 8 стратегиям)")
    if uploaded_files:
        director_res = evaluate_strategy(
            eye_res,
            row,
            contact_established=contact_ok,
            current_date_str=datetime.datetime.now().strftime("%d.%m.%Y"),
        )
        strat_num = director_res["verdict"]["strategy_number"]
        strat_name = director_res["verdict"]["strategy_name"]

        if strat_num in [6, 8]:
            st.error(f"🚨 **{strat_name}**")
        elif strat_num in [2, 3, 4, 5, 7]:
            st.warning(f"⚡ **{strat_name}**")
        else:
            st.success(f"✅ **{strat_name}**")

        st.markdown(f"**🔍 Обоснование ИИ (Explainable AI):**\n> {director_res['explanation']}")
        evidence = director_res.get("evidence") or {}
        if evidence:
            evidence_ru = {
                "strong": "сильная — GPS + QR-маркер полётного задания",
                "medium": "средняя — GPS + EXIF (демо-режим, QR не найден)",
                "weak": "частичная — есть только GPS или только маркер",
                "none": "нет — ни GPS, ни маркера",
            }
            st.caption(
                f"🛡️ **Доказанность снимка:** {evidence_ru.get(evidence.get('level'), evidence.get('level'))} "
                f"· {evidence.get('reason')}"
            )
        st.info(f"✉️ **Черновик SMS / Push клиенту:**\n\n*{director_res['client_message_draft']}*")

        with st.expander("📄 Выходной JSON-контракт AI-Дирижёра", expanded=False):
            st.json(director_res)

        st.markdown("---")
        st.subheader("📝 3. Автоформирование Акта проверки имущества (Приложение 1)")
        st.caption("Готовая карточка для выгрузки в АС «Калита»")

        material_ru = {
            "gas_concrete": "Газобетонные блоки (Сибит)",
            "brick": "Кирпич",
            "wood": "Дерево / профилированный брус",
            "monolith": "Монолитный железобетон",
            "unknown": "Не определен (требуется ручная проверка)"
        }

        with st.form("kalita_act_form"):
            f_col1, f_col2 = st.columns(2)
            act_date = f_col1.text_input("Дата и время осмотра (из EXIF БПЛА):",
                                         value=str(gps.get("date", datetime.datetime.now().strftime("%d.%m.%Y %H:%M"))))
            act_cad = f_col2.text_input("Кадастровый номер ЗУ:", value=str(row["Кадастровый номер"]))
            act_addr = st.text_input("Адрес объекта:", value=str(row["Территория"]))

            m_col1, m_col2 = st.columns(2)
            act_stage = m_col1.text_input("Зафиксированная стадия ИЖС:",
                                          value=eye_res["construction_analysis"]["detected_stage"])
            act_mat = m_col2.text_input("Материал наружных стен:",
                                        value=material_ru.get(eye_res["construction_analysis"]["wall_material"],
                                                              "Газобетон"))

            act_notes = st.text_area("При осмотре установлено (характеристики и дефекты):",
                                     value=eye_res["geospatial_check"]["site_development_notes"], height=100)
            cond_default = "Удовлетворительное" if eye_res["construction_analysis"][
                                                       "detected_stage"] != "0%" else "Неудовлетворительное"
            act_cond = st.selectbox("Состояние недвижимого имущества (визуальный осмотр):",
                                    ["Удовлетворительное", "Неудовлетворительное"],
                                    index=0 if cond_default == "Удовлетворительное" else 1)

            submitted = st.form_submit_button("🚀 Зафиксировать решение и выгрузить Акт в АС «Калита»", type="primary")
            if submitted:
                object_hash = (
                    passport["hash"]["value"]
                    if passport
                    else compute_image_hash(uploaded_files[0])
                )
                try:
                    audit_record = write_audit_log({
                        "actor": "risk-manager-demo",
                        "action": "approve_credit_strategy",
                        "object_hash": object_hash,
                        "decision": f"S{strat_num}: {strat_name}",
                        "details": {
                            "cadastral_number": str(row["Кадастровый номер"]),
                            "client_id": passport["client_id"] if passport else f"CL-{selected_idx + 1:04d}",
                            "flight_task_id": passport["flight_task_id"] if passport else None,
                            "passport_id": passport["passport_id"] if passport else None,
                            "strategy_number": strat_num,
                            "strategy_name": strat_name,
                            "quality_gate": eye_res.get("quality_gate"),
                            "marker_source": meta.get("marker_source"),
                            "ai_source": meta.get("ai", {}).get("source"),
                            "ai_model": meta.get("ai", {}).get("model"),
                            "ai_model_sha256": meta.get("ai", {}).get("model_sha256"),
                            "detected_classes": eye_res["construction_analysis"].get("detected_classes"),
                            "stage_confidence": eye_res["construction_analysis"].get("confidence_stage"),
                        },
                    }, AUDIT_LOG_PATH)
                    if strat_num in [6, 8]:
                        st.warning(
                            "Решение о блокировке автоматики / передаче в СУБ зафиксировано "
                            f"в защищённом Audit Trail (SHA-256: `{audit_record['record_hash'][:16]}…`). "
                            "Акт осмотра в АС «Калита» не выгружается до ручного разбора."
                        )
                    else:
                        st.success(
                            "Акт осмотра успешно отправлен в АС «Калита»! "
                            f"Запись зафиксирована в защищённом Audit Trail (SHA-256: `{audit_record['record_hash'][:16]}…`)."
                        )
                    st.caption(
                        f"record_id: `{audit_record['record_id']}` · "
                        f"hash_prev: `{audit_record['hash_prev'][:16]}…` · "
                        f"head: `{audit_record['record_hash']}`"
                    )
                    updated_audit = audit_chain_report(AUDIT_LOG_PATH)
                    if updated_audit["valid"]:
                        audit_placeholder.success(
                            f"Audit Trail: цепочка цела ({updated_audit['records']} записей)"
                        )
                    else:
                        audit_placeholder.error(
                            f"Audit Trail: нарушение целостности — {updated_audit['error']}"
                        )
                except Exception as audit_error:
                    st.error(f"Не удалось записать событие в Audit Trail: {audit_error}")
    else:
        st.info("👈 Загрузите один или несколько аэрофотоснимков с БПЛА слева, чтобы запустить полный цикл анализа.")
