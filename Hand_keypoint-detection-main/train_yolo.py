from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

model.train(
    data=r"C:\Users\dafer\Desktop\parcial reconocimiento facial\Proyecto_api_ML-Dataset (1)\Proyecto_api_ML-Dataset\dataset.yaml",
    imgsz=420,
    epochs=1,
    batch=8,
)

