from ultralytics import YOLO

# 1. Load the model
model = YOLO("yolo12n.pt") 

# 2. Start training
results = model.train(data="dataset/data.yaml", epochs=100, imgsz=320)

