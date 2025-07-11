from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2

model = YOLO("last.pt")

results = model.predict(source="cracks.mp4", show=True)
print(results)
