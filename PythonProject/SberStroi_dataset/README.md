# SberStroi dataset and training notes

**Источник:** Roboflow-экспорт `hackaton sber.v1i.yolov8.zip` (50 изображений, 10 полётных серий по 5 кадров).

**Классы:** `doors`, `fence`, `foundation`, `roof_finished`, `walls`, `windows`.

## Почему split особенный

В исходном ноутбуке 10 случайных изображений копировались из train в val. Кадры одной серии —
почти одинаковые ракурсы одного объекта, поэтому такая проверка завышает метрики.

`prepare_yolo_split.py` держит в val **целые серии**:

- val: `DJI_20260807142835_0002_D_JPG` (только фундамент → 50%);
- val: `DJI_20260807144205_0005_D_JPG` (кровля + стены + окна + двери → 100%);
- train: остальные 8 серий, 40 изображений.

## Скрипты

| Скрипт | Что делает |
|---|---|
| `prepare_yolo_split.py` | собирает `yolo/` с sequence-level split |
| `train_yolo.py` | обучает holdout-модель 50 эпох и экспортирует ONNX |
| `export_holdout.py` | переэкспортирует именно `best.pt` и пишет честные метрики |
| `train_yolo_all.py` | fine-tune на всех 50 кадрах для стенда (demo-fit, не бенчмарк) |
| `evaluate_stage.py` | считает stage accuracy/confusion на val |

## Важное расхождение

В выводах `SberStroi.ipynb` встречается класс `roof_frame`, а в zip-архиве — `fence`.
Это разные версии датасета. Проект использует классы из zip и одинаковый порядок в
`data.yaml` и `sber_vision.CLASS_NAMES`; не смешивайте веса из ноутбука с этим набором классов.
