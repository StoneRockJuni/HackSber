# Демо для защиты (hackSber)

## 1. Запуск

```powershell
cd C:\Users\Blackview\Desktop\хакатон\PythonProject
.\run.bat
```

Открыть `http://localhost:8501`. В сайдбаре:

- QR-маркер: `STROYKONTUR-DEMO-2026`;
- статус AI: `YOLOv8n SberStroi (ONNX) подключён`;
- статус аудита: цепочка валидна.

## 2. Три бизнес-сценария

| # | Файл | Что показать | Ожидаемо |
|---|---|---|---|
| 1 | `demo_data/demo_object_1_ok.jpg` | нормальный объект, EXIF+GPS+QR, стадия 50% | стратегия 1, плановый мониторинг |
| 2 | `demo_data/demo_object_2_ok.jpg` | дом 100%, не зарегистрирован/не в залоге | стратегия 7, звонок клиенту |
| 3 | `demo_data/demo_fraud_no_exif.jpg` | нет GPS/QR, подозрение на подмену | стратегия 6, стоп и проверка |

После каждого — раскрыть «🔎 Диагностика: откуда GPS и почему такой вердикт».
Там видно `gps_source`, EXIF, SHA-256, источник QR и AI-метаданные.

## 3. Откуда GPS (вопрос жюри)

```powershell
# фото из интернета с EXIF GPS (Италия)
.\tools\inspect_image.py --url "https://raw.githubusercontent.com/ianare/exif-samples/master/jpg/gps/DSCN0010.jpg"

# фото без EXIF
.\tools\inspect_image.py --url "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"

# точная кадастровая точка + QR
.\tools\inspect_image.py --path "C:\photos\house.jpg" --lat 54.8511 --lon 83.1946 --marker STROYKONTUR-DEMO-2026
```

Фразы для ответа:

> GPS берётся только из GPS IFD внутри самого JPEG через Pillow. Если тегов нет — источник `none`.
> Мы не используем IP-геолокацию, карты и данные устройства. Соцсети часто вырезают EXIF,
> поэтому «нет GPS» — это свойство файла, а не сбой приложения.

> `quality_gate_strict` требует QR. `quality_gate_demo_fallback` разрешает EXIF без QR —
> это только демо-послабление, и оно специально показано отдельно.

## 4. AI-сценарий

```powershell
# кадр из val-набора SberStroi: дом с кровлей/стенами/окнами
.\tools\inspect_image.py --path "SberStroi_dataset\yolo\images\val\DJI_20260807144205_0005_D_JPG.rf.7c06371fcc1e87242728b5361392fd0b.jpg"
```

Ожидаемо: стадия `100%`, классы `roof_finished, walls, windows`, высокая уверенность.
На произвольном интернет-фото (Lena, итальянский снимок) — `0%`, `needs_manual_review=true`,
то есть приложение не выдумывает готовность дома.

Фраза для ответа:

> YOLO — это триаж, а не кредитное решение. Если модель не видит двух независимых
> признаков готовности, она отправляет кейс человеку. Это защищает от ложных 100%.

## 5. Проверка аудита

Сначала нажать «Зафиксировать решение», затем:

```powershell
Copy-Item audit_trail.jsonl audit_trail.jsonl.bak
.\tools\inspect_image.py --path demo_data\demo_object_1_ok.jpg --marker STROYKONTUR-DEMO-2026
```

Подменить решение в первой строке (демо-порча):

```powershell
.\.venv\Scripts\python.exe -c "import json,pathlib; p=pathlib.Path('audit_trail.jsonl'); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]; rows[0]['decision']='reject'; p.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')"
.\.venv\Scripts\python.exe -c "from security import verify_audit_chain; print(verify_audit_chain('audit_trail.jsonl'))"
```

Ожидаемо: `False` — цепочка обнаружила изменение. Восстановить:

```powershell
Move-Item audit_trail.jsonl.bak audit_trail.jsonl -Force
```

## 6. Что честно сказать про AI

- holdout-метрики: `mAP50 = 0.344`, `mAP50-95 = 0.149`, stage accuracy `0.6` на 10 val-кадрах;
- это **маленький демо-датасет**, не промышленное качество;
- произвольные фото из интернета — out-of-distribution;
- для пилота нужен полевой датасет, слепая валидация по объектам и мониторинг drift;
- модель не определяет материал стен и не проверяет границы/ВРИ.
