from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

results = model(source=0, show=True, conf=0.85, save=True)