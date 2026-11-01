# SberStroi YOLOv8n — model card

**Артефакт:** `models/sberstroi_yolov8n.onnx`  
**Метаданные:** `models/sberstroi_yolov8n.json`  
**Датасет:** Roboflow-экспорт `hackaton sber.v1i.yolov8.zip`  
**Классы:** `doors`, `fence`, `foundation`, `roof_finished`, `walls`, `windows`

## Как обучалась

- Базовая модель: `yolov8n.pt`.
- 50 эпох, `imgsz=640`, `batch=8`, CPU, seed 42.
- Split **по целым полётным сериям**, а не по отдельным почти одинаковым кадрам:
  - 40 train images (8 серий);
  - 10 val images (2 серии: фундамент и дом с кровлей/окнами/стенами).
- Это меньшее и более честное значение, чем `mAP50≈0.92` из ноутбука: там 10 случайных кадров
  из train попадали в val, то есть модель проверялась на почти тех же снимках.

## Метрики

Смотреть `models/sberstroi_yolov8n.json` (детекция) и `models/sberstroi_stage_eval.json`
(стадийная accuracy на 10 val-кадрах). Это демо-метрики, не промышленный бенчмарк.

## Как принимается стадия

- `100%` — только если есть кровля + стена/окно/дверь, либо окна + двери.
- `50%` — фундамент/стены, либо «только кровля» с низкой уверенностью.
- `0%` — только забор или ничего релевантного, с низкой уверенностью.
- Если уверенность ниже `0.75`, `needs_manual_review=True`, и «Дирижёр» выбирает стратегию 3
  (повторный вылет / инспектор), а не автоматическое решение.

## Известные ограничения

1. **Мало данных.** 40/10 кадров не покрывают регионы, сезоны, ракурсы, материалы и освещение.
2. **False positives на соседних домах.** На кадрах с пустым участком модель может принять
   соседнюю кровлю за `roof_finished`. Поэтому «только кровля» не даёт автоматические 100%.
3. **Нет определения материала стен.** `wall_material` для YOLO-ветки — `unknown`.
4. **Произвольные фото из интернета — out-of-distribution.** Другая камера, ракурс, сжатие,
   рендер или коллаж могут дать случайные детекции; это не проверка качества модели.
5. **Нет adversarial-устойчивости и мониторинга drift.** Для пилота нужны слепая валидация,
   полевая съёмка и периодическая переоценка.

## Воспроизведение

```powershell
python -m venv ml_venv
.\ml_venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\ml_venv\Scripts\python.exe SberStroi_dataset\prepare_yolo_split.py
.\ml_venv\Scripts\python.exe SberStroi_dataset\train_yolo.py
.\.venv\Scripts\python.exe SberStroi_dataset\evaluate_stage.py
```

Runtime-приложение не требует PyTorch/Ultralytics: ONNX читается через `cv2.dnn`.
