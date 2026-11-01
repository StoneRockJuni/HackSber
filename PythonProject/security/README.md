# Security-слой «СтройКонтур»

**Команда:** hackSber  
**Модуль:** `security.py`  
**Документы:** `threat_model.md`, `privacy_152fz.md`, `audit_schema.json`, `sample_passport.json`

## Что проверять на демо

1. **Целостность файла.** `compute_image_hash` считает SHA-256 по всем байтам. Любая правка EXIF или пикселей меняет хэш.
2. **Геопривязка.** `verify_exif_gps_match` сравнивает EXIF GPS с точкой кадастра, по умолчанию допуск 30 м.
3. **Свежесть.** `check_challenge_marker` ищет QR с одноразовым маркером полётного задания. Цветовой маркер — demo-stub.
4. **Паспорт.** `build_snapshot_passport` собирает JSON: хэш, GPS, EXIF, псевдоним клиента, БПЛА, полётное задание, оператор, ссылка верификации, custody.
5. **Аудит.** `write_audit_log` пишет append-only JSONL; каждая запись содержит `hash_prev` и `record_hash`.
6. **Проверка журнала.** `verify_audit_chain` обнаруживает правку или удаление записи в середине цепочки.
7. **Происхождение AI-решения.** Для YOLOv8n SberStroi в `_meta.ai` и в аудит пишутся источник, имя модели, SHA-256 ONNX-файла, найденные классы и уверенность. Если детекций нет, стадия не считается «0%»: включается `needs_manual_review` и стратегия 3.
8. **Происхождение GPS.** `gps_provenance(image, filename)` перечисляет все найденные источники (EXIF GPS IFD → XMP → имя файла), показывает независимые проверки (WGS-84, `GPSStatus`, `0,0`, расхождение EXIF/XMP) и честно возвращает `none`, если координат нет. Отчёт есть в `_meta.gps_provenance` и в `tools/check_photo.py`.
9. **Безопасность ссылок.** Всё скачивание по URL идёт через `net_safe.fetch_image`: только http/https, запрет приватных и локальных адресов после DNS-резолва, проверка каждого редиректа, лимиты и проверка magic bytes (защита от SSRF). Тесты — `tests/test_net_safe.py`.

## Минимальный пример

```python
from security import build_snapshot_passport, verify_audit_chain, write_audit_log

passport = build_snapshot_passport(
    "demo_data/demo_object_1_ok.jpg",
    client_id="CL-2026-004217",
    drone_id="DJI-M3M-NSK-07",
    flight_task_id="FT-2026-10-31-001",
    operator_id="OP-014",
)

record = write_audit_log({
    "actor": "risk-manager-01",
    "action": "approve_credit_strategy",
    "object_hash": passport["hash"]["value"],
    "decision": "S1: плановый мониторинг",
    "details": {
        "client_id": passport["client_id"],
        "flight_task_id": passport["flight_task_id"],
        "passport_id": passport["passport_id"],
    },
}, "audit_trail.jsonl")

assert verify_audit_chain("audit_trail.jsonl")
print(record["record_hash"])
```

## Что честно не сделано

- Нет аутентификации и RBAC в UI.
- Нет аппаратной подписи снимка на БПЛА.
- Нет WORM-хранилища и внешнего якоря аудита.
- Цветовой маркер — только демонстрация; QR не подписан криптографически.
- Автоматического маскирования лиц/номеров нет.
- Внешний VLM не является частью security-контура: для него нужен явный `ALLOW_EXTERNAL_VLM=1`.
- YOLO обучен на крошечном демо-датасете (40 train / 10 val); это не промышленная модель и не гарантия на произвольных интернет-фото.
- Веса ONNX не подписаны; хэш фиксируется в аудите, но подпись артефакта — следующий шаг.
- Нет adversarial-тестов и мониторинга drift модели.

Эти ограничения перечислены в `threat_model.md` и `privacy_152fz.md`, чтобы на защите не выдавать прототип за промышленную СЗИ.
