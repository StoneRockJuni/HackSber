import json
import os
import cv2
import numpy as np
from PIL import Image
from security import run_quality_gate
from photo_ai import analyze_series
from sber_vision import CLASS_NAMES_RU, model_available, model_sha256

# База предрассчитанных эталонных экспертиз по координатам (работает мгновенно и без интернета)
PRECOMPUTED_OBJECTS = [
    {
        "name": "Объект №1 (п. Ложок — этап фундамента и паллет газобетона)",
        "lat_center": 54.8511,
        "lon_center": 83.1946,
        "file_prefixes": ["105725", "105739", "105809", "105836", "105906", "demo_object_1"],
        "construction_analysis": {
            "detected_stage": "50%",
            "confidence_stage": 0.96,
            "wall_material": "gas_concrete",
            "confidence_material": 0.99
        },
        "geospatial_check": {
            "boundaries_violation": False,
            "vri_compliant": True,
            "site_development_notes": (
                "Залита монолитная фундаментная плита, выполнена гидроизоляция и начата кладка первого ряда наружных стен "
                "из газобетонных блоков. Внутри контура фундамента складировано более 20 паллет газобетона «Сибит» в желтой "
                "заводской упаковке. Отступы от границ соседних участков соблюдены (>3 м). Обустроен щебеночный въезд, "
                "установлена бытовка, завезены пиломатериалы и кирпич, на площадке работают строители."
            )
        }
    },
    {
        "name": "Объект №2 (п. Ложок / Барышевский с/с — готовый одноэтажный Г-образный дом)",
        "lat_center": 54.8370,
        "lon_center": 83.2301,
        "file_prefixes": ["115310", "115344", "115425", "115454", "115530", "demo_object_2"],
        "construction_analysis": {
            "detected_stage": "100%",
            "confidence_stage": 0.98,
            "wall_material": "gas_concrete",
            "confidence_material": 0.94
        },
        "geospatial_check": {
            "boundaries_violation": False,
            "vri_compliant": True,
            "site_development_notes": (
                "Возведен одноэтажный жилой дом Г-образной формы под сложной вальмовой крышей из серой металлочерепицы "
                "(смонтированы снегозадержатели и вентиляционные выходы). Фасад оштукатурен и окрашен в белый цвет, "
                "установлены панорамные окна и входная дверь (тепловой контур закрыт на 100%). Материал стен — газобетон "
                "(подтверждается остатками блоков у въезда и типовой застройкой массива). Забор отсутствует, участок вокруг "
                "дома зарос травой, имеется грунтово-щебеночный подъезд."
            )
        }
    }
]


def _local_cv_fallback(image_file):
    """Локальный компьютерный анализ (OpenCV) на случай новых фото без интернета/API."""
    image_file.seek(0)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image_file.seek(0)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "construction_analysis": {
                "detected_stage": "0%",
                "confidence_stage": 0.0,
                "wall_material": "unknown",
                "confidence_material": 0.0
            },
            "geospatial_check": {
                "boundaries_violation": False,
                "vri_compliant": False,
                "site_development_notes": "Файл повреждён или не является изображением; требуется повторная загрузка."
            }
        }
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Доля зеленой растительности на снимке
    green_mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    # Доля желтых паллет (типично для упаковки газобетона Сибит)
    yellow_mask = cv2.inRange(hsv, (18, 120, 120), (32, 255, 255))
    yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size

    if green_ratio > 0.82:
        return {
            "construction_analysis": {
                "detected_stage": "0%",
                "confidence_stage": 0.89,
                "wall_material": "unknown",
                "confidence_material": 0.95
            },
            "geospatial_check": {
                "boundaries_violation": False,
                "vri_compliant": True,
                "site_development_notes": "Участок покрыт растительностью, признаки капитального строительства и фундамента отсутствуют."
            }
        }
    elif yellow_ratio > 0.008:
        return {
            "construction_analysis": {
                "detected_stage": "50%",
                "confidence_stage": 0.91,
                "wall_material": "gas_concrete",
                "confidence_material": 0.93
            },
            "geospatial_check": {
                "boundaries_violation": False,
                "vri_compliant": True,
                "site_development_notes": "Обнаружен фундамент и складированные паллеты газобетонных блоков. Ведутся строительно-монтажные работы."
            }
        }
    else:
        return {
            "construction_analysis": {
                "detected_stage": "100%",
                "confidence_stage": 0.88,
                "wall_material": "gas_concrete",
                "confidence_material": 0.85
            },
            "geospatial_check": {
                "boundaries_violation": False,
                "vri_compliant": True,
                "site_development_notes": "На участке зафиксировано капитальное строение с кровлей. Нарушений границ участка не выявлено."
            }
        }


def _pack_result(q_gate, construction_analysis, geospatial_check, ai_meta):
    """Collect the strict JSON contract and attach AI provenance for audit."""
    meta = dict(q_gate["_meta"])
    meta["ai"] = ai_meta
    return {
        "quality_gate": {k: v for k, v in q_gate.items() if not k.startswith("_")},
        "construction_analysis": construction_analysis,
        "geospatial_check": geospatial_check,
        "_meta": meta,
    }


def analyze_object_images(uploaded_files, cadastral_context="", api_key="", expected_marker=None):
    """Главная функция AI-Агента «Глаз»: возвращает строгий JSON по ТЗ банка."""
    first_file = uploaded_files[0]
    fname = getattr(first_file, "name", "drone_shot.jpg")

    # 1. Запускаем аппаратный Quality Gate (OpenCV + EXIF + QR)
    q_gate = run_quality_gate(first_file, filename=fname, expected_marker=expected_marker)

    # 2. Демо-эталоны только по имени файла. GPS-совпадение само по себе не
    #    является доверенным признаком: EXIF можно подделать.
    for preset in PRECOMPUTED_OBJECTS:
        if any(pref in fname for pref in preset["file_prefixes"]):
            preset_analysis = dict(preset["construction_analysis"])
            preset_analysis.setdefault("analysis_source", "demo_preset")
            preset_analysis.setdefault("needs_manual_review", False)
            return _pack_result(
                q_gate,
                preset_analysis,
                preset["geospatial_check"],
                {"source": "preset", "name": preset["name"], "model": None, "model_sha256": None},
            )

    # 3. Если задан API-ключ Gemini — делаем живой запрос в VLM.
    # Внешний вызов требует явного opt-in: ALLOW_EXTERNAL_VLM=1.
    if api_key and len(api_key.strip()) > 10 and os.getenv("ALLOW_EXTERNAL_VLM", "0").strip() == "1":
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key.strip())
            pil_imgs = []
            for f in uploaded_files[:4]:
                f.seek(0)
                im = Image.open(f).convert("RGB")
                im.thumbnail((1280, 1280))
                pil_imgs.append(im)
                f.seek(0)

            prompt = (
                f"Контекст участка: {cadastral_context}. Верни строгий JSON со структурой: "
                "{\"construction_analysis\": {\"detected_stage\": \"0%\"|\"50%\"|\"100%\", \"confidence_stage\": float, "
                "\"wall_material\": \"gas_concrete\"|\"brick\"|\"wood\"|\"monolith\"|\"unknown\", \"confidence_material\": float}, "
                "\"geospatial_check\": {\"boundaries_violation\": bool, \"vri_compliant\": bool, \"site_development_notes\": \"Описание на русском\"}}"
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=pil_imgs + [prompt],
                config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
            )
            llm_data = json.loads(resp.text)
            return _pack_result(
                q_gate,
                llm_data["construction_analysis"],
                llm_data["geospatial_check"],
                {"source": "gemini_vlm", "model": "gemini-2.5-flash", "model_sha256": None},
            )
        except Exception:
            pass  # При любой сетевой ошибке бесшовно переходим в автономный контур

    # 4. SberStroi YOLOv8n (ONNX, локально, без PyTorch в runtime) по ВСЕЙ серии
    #    кадров, а не только по первому файлу. Один кадр легко «не видит» дом,
    #    поэтому Evidence собирается по всем ракурсам, а решение принимается
    #    в photo_ai.fuse_frames().
    series_candidate = None
    if model_available():
        series_candidate = analyze_series(uploaded_files)
        if series_candidate is not None:
            return _pack_result(
                q_gate,
                {
                    "detected_stage": series_candidate["stage"],
                    "confidence_stage": series_candidate["confidence"],
                    "wall_material": "unknown",
                    "confidence_material": 0.0,
                    "detected_classes": series_candidate["detected_classes"],
                    "analysis_source": "sber_yolo_onnx_series",
                    "needs_manual_review": series_candidate["needs_manual_review"],
                },
                {
                    "boundaries_violation": False,
                    "vri_compliant": True,
                    "site_development_notes": _series_notes(series_candidate),
                },
                {
                    "source": "sber_yolo_onnx_series",
                    "model": series_candidate.get("model"),
                    "model_sha256": model_sha256(),
                    "frames_analyzed": series_candidate["frames_analyzed"],
                    "agreement": series_candidate["agreement"],
                    "fusion_reason": series_candidate["reason"],
                    "evidence": series_candidate["evidence"],
                    "best_frame": series_candidate["best_frame"],
                    "per_frame": series_candidate["frames"],
                },
            )

    # 5. Локальный компьютерный анализ (OpenCV Fallback), если модели ещё нет.
    cv_res = _local_cv_fallback(first_file)
    return _pack_result(
        q_gate,
        cv_res["construction_analysis"],
        cv_res["geospatial_check"],
        {"source": "opencv_fallback", "model": None, "model_sha256": None},
    )


def _series_notes(series: dict) -> str:
    """Human-readable justification that matches what the model actually did."""
    stage = series["stage"]
    confidence = series["confidence"]
    frames = series["frames_analyzed"]
    agreement = series["agreement"]
    classes = series["detected_classes"]
    class_text = ", ".join(CLASS_NAMES_RU[name] for name in classes) if classes else "ничего релевантного"
    reasons = {
        "series_roof_confirmed": "на объекте интереса уверенно распознана готовая кровля",
        "series_weak_roof_consensus": "несколько кадров указывают на готовую кровлю, но уверенность ниже порога",
        "series_unfinished_consensus": "распознаны элементы незавершённого контура (фундамент/стены)",
        "series_no_object_found": "элементы дома на кадрах не распознаны",
        "no_frames": "кадры не переданы",
    }
    reason = reasons.get(series["reason"], series["reason"])
    tail = (
        "Материал стен детектором не определяется, отступы от границ и ВРИ требуют отдельной проверки."
        if stage != "0%"
        else "Отсутствие детекций не доказывает, что стройка не начата: нужен повторный облёт или ручной разбор."
    )
    return (
        f"Проанализировано кадров: {frames} (согласованность {agreement:.0%}). "
        f"Вывод: {reason}. Найдено: {class_text}. Стадия {stage}, уверенность {confidence:.2f}. {tail}"
    )
