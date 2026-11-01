import datetime

from security import evidence_level

# YOLO/fuzzy decisions below this confidence are not auto-approved. This is
# deliberately above the confidence of walls/foundation-only stage 50.
AI_MANUAL_REVIEW_CONFIDENCE = 0.75


def evaluate_strategy(eye_json: dict, registry_row: dict, contact_established: bool = True, current_date_str: str | None = None) -> dict:
    """Выбрать стратегию и всегда приложить уровень доказанности снимка."""
    result = _decide_strategy(eye_json, registry_row, contact_established, current_date_str)
    result["evidence"] = evidence_level(eye_json["quality_gate"], eye_json.get("_meta"))
    return result


def _decide_strategy(eye_json: dict, registry_row: dict, contact_established: bool = True, current_date_str: str | None = None) -> dict:
    """
    Принимает JSON от AI-Агента «Глаз» и строку из Реестра_для_теста.xlsx.
    Возвращает строгий JSON AI-Дирижёра с выбранной стратегией (1-8), объяснением и черновиком SMS клиенту.
    """
    if not current_date_str:
        current_date_str = datetime.datetime.now().strftime("%d.%m.%Y")
    q_gate = eye_json["quality_gate"]
    c_analysis = eye_json["construction_analysis"]
    geo_check = eye_json["geospatial_check"]

    stage = c_analysis["detected_stage"]
    has_pz = str(registry_row.get("Наличие ПЗ", "Нет")).strip().lower() == "да"
    is_registered = str(registry_row.get("ОН зарегистрирован", "Нет")).strip().lower() == "да"
    is_pledged = str(registry_row.get("Залог", "Нет")).strip().lower() == "да"

    # Расчет оставшегося срока строительства по датам из реестра
    try:
        cur_dt = datetime.datetime.strptime(current_date_str, "%d.%m.%Y")
        plan_dt = datetime.datetime.strptime(str(registry_row.get("Плановая дата", "01.01.2027")).strip(), "%d.%m.%Y")
        start_dt = datetime.datetime.strptime(str(registry_row.get("Дата начала строительства", "01.01.2025")).strip(), "%d.%m.%Y")
        total_days = max((plan_dt - start_dt).days, 1)
        remaining_days = (plan_dt - cur_dt).days
        remaining_ratio = remaining_days / total_days
        deadline_expired = remaining_days < 0
    except Exception:
        remaining_days = 180
        remaining_ratio = 0.5
        deadline_expired = False

    # 1. Проверка подлинности снимка — САМАЯ ПЕРВАЯ.
    #
    # Дрон НЕ подписывает снимки, а на части кадров метаданных нет вообще, поэтому
    # «нет GPS» само по себе не означает фрод. Смотрим на уровень доказанности:
    #   strong  - GPS + настоящий QR-маркер  -> обычный маршрут;
    #   medium  - GPS + EXIF (демо-режим)    -> обычный маршрут с пометкой;
    #   weak    - что-то одно                -> ручная проверка (стратегия 3);
    #   none    - ничего нет                 -> риск фрода (стратегия 6).
    #
    # Порядок важен: если файл вообще нельзя привязать к объекту залога (чертёж,
    # скриншот, картинка из интернета), то «повторный вылет» - неправильный ответ:
    # летать не за чем, пока непонятно, тот ли это объект. Такие файлы одинаково
    # уходят в СУБ, независимо от резкости и яркости.
    evidence = evidence_level(q_gate, eye_json.get("_meta"))
    if evidence["level"] == "none":
        return {
            "verdict": {"strategy_number": 6, "strategy_name": "СТРАТЕГИЯ 6: Блокировка автоматики / Передача в Службу безопасности (СУБ), риск фрода"},
            "evidence": evidence,
            "explanation": (
                "Снимок невозможно привязать к объекту залога: нет ни GPS, ни QR-маркера "
                "полётного задания. Это может быть чертёж, скриншот или чужое фото — "
                "поэтому автоматика блокируется, а материал уходит в СУБ."
            ),
            "client_message_draft": "Выдача очередного транша по кредитному договору приостановлена. Просим связаться с менеджером банка.",
        }

    # 2. Проверка качества снимка (только для снимков, которые уже привязаны
    #    к объекту: плохой кадр реального объекта -> нужен повторный вылет).
    if not q_gate["image_quality_passed"]:
        return {
            "verdict": {"strategy_number": 3, "strategy_name": "СТРАТЕГИЯ 3: Запрос повторного вылета / Выезд инспектора"},
            "explanation": (
                f"Снимок не прошел контроль качества (Quality Gate): {q_gate['error_message']}. "
                "Объект к съёмке привязан, но кадр непригоден для оценки — нужен повторный пролёт."
            ),
            "client_message_draft": "Уважаемый клиент! В рамках планового мониторинга ИЖС потребуется повторная аэрофотосъемка участка."
        }

    if evidence["level"] == "weak":
        return {
            "verdict": {"strategy_number": 3, "strategy_name": "СТРАТЕГИЯ 3: Запрос повторного вылета / Выезд инспектора"},
            "evidence": evidence,
            "explanation": (
                f"Снимок не доказан полностью: {evidence['reason']}. "
                "Это не фрод, но и не основание для автоматического решения — "
                "нужен повторный кадр с маркером полётного задания или выезд инспектора."
            ),
            "client_message_draft": "Уважаемый клиент! Для подтверждения осмотра потребуется повторная аэрофотосъёмка участка.",
        }

    # 3. Нарушение границ кадастра или ВРИ
    if geo_check["boundaries_violation"] or not geo_check["vri_compliant"]:
        return {
            "verdict": {"strategy_number": 4, "strategy_name": "СТРАТЕГИЯ 4: Замена обеспечения"},
            "explanation": "Зафиксировано нарушение отступов от кадастровых границ участка или несоответствие ВРИ.",
            "client_message_draft": "При осмотре объекта ИЖС выявлено отклонение от градостроительных норм посадки здания. Требуется согласование обеспечения."
        }

    # 3.1. Недостаточная уверенность AI: не превращаем "не видно" в "0%".
    stage_confidence = float(c_analysis.get("confidence_stage", 1.0))
    needs_review = bool(c_analysis.get("needs_manual_review")) or stage_confidence < AI_MANUAL_REVIEW_CONFIDENCE

    # Исключение: снимок доказан (GPS + одноразовый QR-маркер полётного задания).
    # Тогда «объекта не видно» - это факт про участок, а не сомнение в кадре:
    # подтверждено, что снят нужный объект и снят сейчас. Такой достоверный 0%
    # уходит в кредитную логику (интенсивный мониторинг / взыскание), а не в
    # бесконечный «повторный вылет».
    proven_empty = stage == "0%" and evidence["level"] == "strong"
    if proven_empty:
        needs_review = False

    if needs_review:
        if stage == "0%":
            explanation = (
                f"На кадрах не видно объекта залога (уверенность {stage_confidence:.2f}, "
                f"источник: {c_analysis.get('analysis_source', 'unknown')}). "
                "Блокировать кредит на основании «ничего не видно» нельзя: кадр мог быть снят "
                "не над тем участком, объект мог остаться за краем кадра, а дрон не подписывает "
                "снимки. Сначала повторный облёт с QR-маркером или выезд инспектора — "
                "и только подтверждённое «работы не начаты» включает взыскание."
            )
        else:
            explanation = (
                f"AI-Агент не смог уверенно определить стадию объекта "
                f"(уверенность {stage_confidence:.2f}, источник: {c_analysis.get('analysis_source', 'unknown')}). "
                "Автоматическое кредитное решение не принимается."
            )
        return {
            "verdict": {"strategy_number": 3, "strategy_name": "СТРАТЕГИЯ 3: Запрос повторного вылета / Выезд инспектора"},
            "explanation": explanation,
            "client_message_draft": "Уважаемый клиент! Для подтверждения готовности объекта потребуется повторная аэрофотосъемка или выезд инспектора."
        }

    # 4. Если объект построен на 100%, но еще не оформлен в залог Банка (Ветка 7 на блок-схеме)
    if stage == "100%" and (not is_registered or not is_pledged):
        return {
            "verdict": {"strategy_number": 7, "strategy_name": "СТРАТЕГИЯ 7: Регистрация ОН и оформление в залог Банка"},
            "explanation": f"AI-Агент зафиксировал 100% готовность дома, однако в реестре объект не зарегистрирован ({registry_row.get('ОН зарегистрирован')}) и не оформлен в залог ({registry_row.get('Залог')}).",
            "client_message_draft": "Поздравляем с завершением строительства дома! Пожалуйста, поставьте объект на кадастровый учет и предоставьте выписку ЕГРН для оформления залога и сохранения ставки."
        }

    # 5. Если есть просроченная задолженность (ПЗ == Да) или плановая дата истекла при незавершенной стройке
    if has_pz and deadline_expired and stage == "0%":
        return {
            "verdict": {"strategy_number": 8, "strategy_name": "СТРАТЕГИЯ 8: Обращение взыскания и реализация ОН (Торги PortalDA)"},
            "explanation": f"Плановая дата строительства ({registry_row.get('Плановая дата')}) истекла, имеется просроченная задолженность (ПЗ: Да), работы на участке не начинались (0%).",
            "client_message_draft": "Уведомляем о начале процедуры реализации залогового имущества в связи с неисполнением условий кредитного договора."
        }

    if has_pz:
        if contact_established and not deadline_expired:
            return {
                "verdict": {"strategy_number": 2, "strategy_name": "СТРАТЕГИЯ 2: Реструктуризация / Урегулирование"},
                "explanation": f"По кредиту зафиксирована просроченная задолженность (ПЗ: Да), но строительство ведется (стадия {stage}), клиент идет на контакт.",
                "client_message_draft": "Сибирский банк предлагает рассмотреть инструменты реструктуризации для комфортного завершения строительства вашего дома."
            }
        else:
            return {
                "verdict": {"strategy_number": 6, "strategy_name": "СТРАТЕГИЯ 6: Передача в суд (Судебное взыскание)"},
                "explanation": f"Наличие просроченной задолженности (ПЗ: Да) при истечении плановых сроков ({registry_row.get('Плановая дата')}) и отсутствии урегулирования.",
                "client_message_draft": "Срочное уведомление: в связи с просроченной задолженностью и нарушением сроков строительства материалы передаются в судебное производство."
            }

    # 6. Просрочки нет (ПЗ == Нет), но плановая дата уже прошла, а дом не достроен (<100%)
    if deadline_expired and stage != "100%":
        if contact_established:
            return {
                "verdict": {"strategy_number": 3, "strategy_name": "СТРАТЕГИЯ 3: Доп. финансирование на достройку ОН / Пересмотр графика"},
                "explanation": f"Плановая дата ({registry_row.get('Плановая дата')}) истекла, текущая готовность — {stage}. Просрочек по платежам нет, клиент на связи.",
                "client_message_draft": "Плановый срок завершения строительства истек. Предлагаем продлить график строительства или оформить доп. финансирование для завершения работ."
            }
        else:
            return {
                "verdict": {"strategy_number": 4, "strategy_name": "СТРАТЕГИЯ 4: Замена обеспечения"},
                "explanation": f"Плановый срок строительства ({registry_row.get('Плановая дата')}) истек, объект не достроен ({stage}), коммуникация с клиентом неуспешна.",
                "client_message_draft": "В связи с незавершением строительства в установленный договором срок просим срочно связаться с Банком по вопросу замены залогового обеспечения."
            }

    # 7. Стройка в рамках срока, но осталось менее 30% времени при низкой готовности
    if remaining_ratio <= 0.30 and stage != "100%":
        return {
            "verdict": {"strategy_number": 5, "strategy_name": "СТРАТЕГИЯ 5: Интенсивный мониторинг (Риск срыва срока)"},
            "explanation": f"До плановой даты завершения ({registry_row.get('Плановая дата')}) осталось менее 30% срока ({remaining_days} дн.), текущая готовность — {stage}.",
            "client_message_draft": f"Напоминаем, что плановая дата завершения строительства вашего дома — {registry_row.get('Плановая дата')}. Просим ускорить темпы строительных работ."
        }

    # 7.1. Достоверно зафиксировано 0%: объект не начат. Это НЕ «стройка идёт по
    # графику» и не повод молча ждать. Включаем интенсивный мониторинг сразу,
    # не дожидаясь, пока до плановой даты останется 30% срока. Взыскание (8)
    # включается только при просрочке и истёкшем сроке - это выше.
    if stage == "0%":
        return {
            "verdict": {"strategy_number": 5, "strategy_name": "СТРАТЕГИЯ 5: Интенсивный мониторинг (Риск срыва срока)"},
            "explanation": (
                f"Работы на участке достоверно не начаты (стадия 0%, уверенность "
                f"{stage_confidence:.2f}): нет фундамента, стен и кровли. Просрочки по платежам нет "
                f"и плановая дата ({registry_row.get('Плановая дата')}) не истекла, поэтому сразу "
                "взыскание не запускается — кредит переводится на интенсивный мониторинг."
            ),
            "client_message_draft": f"При дистанционном осмотре не зафиксировано начало строительных работ. Плановая дата завершения — {registry_row.get('Плановая дата')}. Просим сообщить о планах по началу работ."
        }

    # 8. Штатный ход строительства (Стратегия 1)
    return {
        "verdict": {"strategy_number": 1, "strategy_name": "СТРАТЕГИЯ 1: Плановый мониторинг (Вылет БПЛА)"},
        "explanation": f"Строительство идет в соответствии с графиком (текущая стадия {stage}, плановая дата {registry_row.get('Плановая дата')}, просроченной задолженности нет).",
        "client_message_draft": f"Дистанционный осмотр вашего участка ИЖС прошел успешно (зафиксирована стадия {stage}). Следующий транш одобрен!"
    }
