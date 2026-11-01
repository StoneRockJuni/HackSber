# СтройКонтур — hackSber

Прототип дистанционного мониторинга объектов ИЖС по аэрофотоснимкам БПЛА.
В проекте три слоя:

1. **«Глаз»** — качество снимка, EXIF/GPS, QR-маркер, локальный анализ и нейросеть.
2. **«AI-Дирижёр»** — выбор одной из 8 кредитных стратегий и черновик сообщения клиенту.
3. **Security** — паспорт снимка, журнал аудита с hash chain, GPS-provenance,
   документы по угрозам и 152-ФЗ.

## Быстрый запуск на Windows

В проекте уже создано рабочее окружение `.venv` (Python 3.13). Старая macOS-папка `.venv` заменена: на Windows она не запускается.

Самый простой способ:

```bat
run.bat
```

Или вручную из PowerShell:

```powershell
cd C:\Users\Blackview\Desktop\хакатон\PythonProject
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Если окружение нужно пересоздать:

```powershell
python3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

> `python` из MSYS2 (3.14) для этого проекта не подходит: у него нет pip в нужной конфигурации и он ломает сборку numpy/opencv. Используйте Python 3.13 из Microsoft Store или python.org.

Если открываете проект в PyCharm: **Settings → Python Interpreter → Add Interpreter → Existing** и укажите `.venv\Scripts\python.exe`. Старая папка `.idea` от macOS переименована в `.idea-macos-backup`, чтобы PyCharm не искал несуществующий Mac-интерпретатор.

## Демо-сценарии

В папке `demo_data` лежат три синтетических снимка. Они не содержат реальных ПДн.

| Файл | Что показывает | Ожидаемая стратегия |
|---|---|---|
| `demo_object_1_ok.jpg` | EXIF + GPS 54.8511/83.1946 + QR-маркер, стадия 50% | Стратегия 1 — плановый мониторинг |
| `demo_object_2_ok.jpg` | EXIF + GPS 54.8370/83.2301 + QR, дом 100% | Стратегия 7 — регистрация ОН и залог |
| `demo_fraud_no_exif.jpg` | Качественный кадр без EXIF/GPS и без QR | Стратегия 6 — риск фрода, блокировка автоматики |

Порядок демонстрации:

1. Выберите объект в боковой панели.
2. Загрузите снимок из `demo_data`.
3. Проверьте блок «Паспорт снимка / Chain of Custody»: SHA-256, GPS, QR, псевдоним клиента, ссылка верификации.
4. Нажмите «Зафиксировать решение…» — в `audit_trail.jsonl` появится запись.
5. В боковой панели нажмите «Проверить целостность журнала».
6. Для демонстрации подмены измените символ в `audit_trail.jsonl` и снова нажмите проверку — цепочка станет невалидной.

Пересоздать демо-снимки:

```powershell
.\.venv\Scripts\python.exe demo_data\generate_demo_images.py
```

## AI-слой SberStroi (YOLOv8n) — версия 2: серия кадров вместо одного фото

В `SberStroi.ipynb` была обучена YOLOv8n на 6 классах: `doors`, `fence`, `foundation`,
`roof_finished`, `walls`, `windows`. Логика перенесена в проект и переработана:

| Модуль | Роль |
|---|---|
| `sber_vision.py` | ONNX-инференс через `cv2.dnn` (без PyTorch), кэш сети, декодирование выходов |
| `photo_ai.py` | **объект-центричный анализ кадра + слияние по серии кадров** (главное изменение) |
| `vision_agent.py` | склейка: Quality Gate → демо-пресеты → VLM (off) → YOLO по всей серии → OpenCV fallback |
| `director.py` | 8 кредитных стратегий; при `confidence < 0.75` — стратегия 3 (повторный вылет/инспектор) |

### Что исправлено по сравнению с ноутбуком

1. **Анализируется вся серия кадров**, а не первый файл. Раньше `uploaded_files[1:]`
   показывались в превью и не участвовали в решении.
2. **Кровля в центре кадра = 100 %.** С высоты «в надир» у готового дома видно
   в основном кровлю, окон и дверей сверху нет. Старое правило требовало
   «кровля + стена/окно/дверь» и записывало готовые дома в 50 %.
3. **Соседская кровля больше не даёт 100 %**: детекция у края кадра считается
   фоном и не используется в решении.
4. **«Ничего не найдено» ≠ уверенные 0 %**: при низкой уверенности возвращается
   `needs_manual_review`, и «Дирижёр» выбирает стратегию 3.
5. **Кэш ONNX-сети**: серия из 24 кадров обрабатывается ≈11 с, а не парсит
   13 МБ модели на каждый кадр.

### Замеренное поведение

| Логика | Результат на 56 реальных кадрах (13 вылетов) |
|---|---|
| Старая: решение по первому кадру | 4 из 13 вылетов готовых домов получили 50 % вместо 100 % |
| Наивная: объединить классы всех кадров | ложные 100 % на стройплощадке (соседская кровля) |
| Новая: объект-центричное слияние | 12 из 13 — ожидаемая стадия, 1 — честный промах → инспектор |

Отдельно исправлено после проверки на реальной загрузке: **одиночное фото
готового дома** больше не уходит в «повторный вылет» с уверенностью 0.00. Раньше
слабый (но найденный) кадр со 100 % проваливался в ветку «0 %», а формула
уверенности была слишком пессимистичной для кадров 5280×3956.

Метрики на размеченном сплите, честность цифр и протокол валидации —
[`docs/AI_VALIDATION.md`](docs/AI_VALIDATION.md).

Примеры разметки на реальных кадрах:
[готовый дом — 100 %](docs/gallery/finished_100pct.jpg) ·
[стройплощадка — 50 % и отказ](docs/gallery/foundation_50pct.jpg).

Переобучение (отдельное окружение, CPU):

```powershell
python -m venv ml_venv
.\ml_venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\ml_venv\Scripts\python.exe SberStroi_dataset\prepare_yolo_split.py
.\ml_venv\Scripts\python.exe SberStroi_dataset\train_yolo.py
```

Важно: демо-датасет **40 train / 10 val**, split по целым полётным сериям.
Метрики из JSON нельзя выдавать за промышленное качество; произвольные фото из
интернета — out-of-distribution. Для пилота нужен полевой датасет, слепая
валидация по объектам и мониторинг drift.

Переопределить путь к модели:

```powershell
$env:SBER_YOLO_MODEL = "D:\models\sberstroi_v2.onnx"
```

## Проверка любого фото (с диска, из папки или из интернета)

`tools/check_photo.py` — один инструмент на все вопросы: откуда GPS, прошёл ли
Quality Gate, что видит нейросеть, что с этим делать.

```powershell
# локальный файл + размеченный кадр
.\.venv\Scripts\python.exe tools\check_photo.py --path "C:\photos\house.jpg" --overlay out.jpg

# файл по URL (скачивается локально, с SSRF-фильтром)
.\.venv\Scripts\python.exe tools\check_photo.py --url "https://example.com/photo.jpg"

# серия кадров одного объекта
.\.venv\Scripts\python.exe tools\check_photo.py --dir "..\домики" --overlay out.jpg --json
```

Что важно понимать:

- GPS берётся **только из самого файла**: EXIF GPS IFD → XMP-пакет → имя файла
  вида `54.8511_83.1946.jpg`. Приложение не использует IP-геолокацию, карты или
  данные устройства.
- Если фото скачано из соцсети/мессенджера, EXIF часто уже удалён. Тогда
  `gps_source = none` — это свойство файла, а не сбой проверки.
- `quality_gate_strict` — честный режим: QR/маркер обязателен.
- `quality_gate_demo_fallback` — демо-режим приложения: EXIF без QR тоже
  считается сигналом свежести.

Ещё два инструмента:

```powershell
# инспектор только по security-слою (EXIF, сырые GPS-теги, паспорт)
.\.venv\Scripts\python.exe tools\inspect_image.py --path "C:\photos\house.jpg" --passport

# честная оценка модели на размеченном наборе (YOLO-папка или CSV-манифест)
.\.venv\Scripts\python.exe tools\eval_model.py --yolo SberStroi_dataset\yolo\images\val --labels SberStroi_dataset\yolo\labels\val
.\.venv\Scripts\python.exe tools\eval_model.py --manifest my_test.csv --manifest-out report.json
```

## «Песочница» в интерфейсе

Кроме основного банковского сценария (`app.py`) есть отдельная страница
`pages/1_🔬_Песочница.py`:

- загрузка одного фото, серии или **ссылки из интернета**;
- источник GPS + независимые проверки метаданных;
- Quality Gate (резкость, геопривязка, QR);
- рамки детектора и зона «объект интереса» на картинке;
- вердикт «авто / к инспектору».

Запуск тот же: `run.bat` → в сайдбаре появится страница «Песочница».

## Security-модуль

Основные функции в `security.py`:

- `compute_image_hash(path)` — SHA-256 по всем байтам файла.
- `gps_provenance(image, filename)` — откуда взялись координаты: EXIF → XMP →
  имя файла, плюс независимые проверки метаданных.
- `evidence_level(quality_gate, meta)` — уровень доказанности снимка
  (`strong` / `medium` / `weak` / `none`). Дрон не подписывает снимки, поэтому
  «нет GPS» — не фрод, а ручная проверка.
- `verify_exif_gps_match(image_path, cadastral_gps, tolerance_m=30)` — сверка GPS с кадастровой точкой.
- `check_challenge_marker(image_path, expected_marker)` — проверка QR (и demo-цветового маркера).
- `build_snapshot_passport(...)` — паспорт снимка (chain of custody).
- `write_audit_log(event, log_path)` — append-only JSONL с `hash_prev` и `record_hash`.
- `verify_audit_chain(log_path)` — проверка всей цепочки.

Безопасность исходящих запросов — `net_safe.py`: только http/https, запрет
приватных и локальных адресов, проверка каждого редиректа, лимиты и проверка
magic bytes (защита от SSRF при проверке фото по ссылке).

Переменные окружения (секретов в коде нет):

```powershell
$env:PASSPORT_VERIFY_BASE_URL = "https://bank.example/verify"
$env:PASSPORT_SIGNING_KEY = "..."   # включит HMAC-SHA256 подпись паспорта
$env:AUDIT_LOG_PATH = "audit_trail.jsonl"
$env:ALLOW_EXTERNAL_VLM = "1"       # только для демо, не для реальных ПДн
```

Автотесты:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

> В среде с урезанными правами на `%TEMP%` часть тестов, использующих
> `tempfile`, может падать с `PermissionError` — это ограничение окружения, а не
> кода: запускайте из обычного терминала.

## Документы

- [`docs/HOW_IT_WORKS_SIMPLE.md`](docs/HOW_IT_WORKS_SIMPLE.md) — **объяснение
  для человека без опыта**: аналогии, «как ребёнок учится», роли, частые вопросы.
- [`docs/EXPLAINER.md`](docs/EXPLAINER.md) — **весь проект простыми словами**:
  кейс, роли, каждый модуль, что говорят на защите, словарь терминов.
- [`docs/UI_GUIDE.md`](docs/UI_GUIDE.md) — **разбор интерфейса до мелочи**:
  откуда каждая цифра KPI, что значит ПЗ, что делает каждая кнопка.
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — что изменилось в этой итерации,
  до/после, и почему обучение в Colab, а продукт локально.
- [`docs/QA_BANK.md`](docs/QA_BANK.md) — **135 вопросов и ответов** для защиты
  (безопасность, ИИ, бизнес, техника, неудобные).
- [`docs/SURVEY_PROTOCOL.md`](docs/SURVEY_PROTOCOL.md) — протокол съёмки:
  высоты, перекрытие, GSD, периодичность, полётное задание в JSON.
- [`docs/ECONOMICS.md`](docs/ECONOMICS.md) — экономика: расчёт трудозатрат,
  формула денег и что мерить в пилоте.
- [`docs/QR_MARKER_DESIGN.md`](docs/QR_MARKER_DESIGN.md) — QR-маркер полётного
  задания: как должно работать, формат, размеры печати, угрозы.
- [`docs/AI_VALIDATION.md`](docs/AI_VALIDATION.md) — как проверять нейросеть,
  почему фото из интернета не измеряют точность и отличит ли модель готовность.
- [`docs/SECURITY_REPORT.md`](docs/SECURITY_REPORT.md) — откуда GPS, что
  защищено, SSRF-защита, уровень доказанности снимка, план до пилота.
- [`security/threat_model.md`](security/threat_model.md) — модель угроз T1–T20.
- [`security/privacy_152fz.md`](security/privacy_152fz.md) — ПДн, роли, сроки.
- [`security/audit_schema.json`](security/audit_schema.json) — JSON Schema аудита.
- [`security/sample_passport.json`](security/sample_passport.json) — пример паспорта.
- [`security/README.md`](security/README.md) — карта security-слоя для защиты.
- [`models/README.md`](models/README.md) — model card: как обучалась, метрики, ограничения.

## Важно

- Внешний Gemini VLM отключён по умолчанию: даже при введённом ключе нужен явный `ALLOW_EXTERNAL_VLM=1`. Не отправляйте через него реальные снимки и ПДн.
- Прототип не содержит промышленной аутентификации/RBAC: это следующий шаг до пилота.
- Журнал аудита защищает от правки/удаления в середине цепочки, но не от удаления всего файла. Для пилота нужен WORM-регистр и внешний якорь head-хэша.
- «Песочница» не заменяет валидацию модели: интернет-фото — out-of-distribution,
  см. `docs/AI_VALIDATION.md`.
