from ultralytics import YOLO

# 1. Load your custom YOLOv12 model
# This uses the specific weights path you provided
model = YOLO('runs/detect/train/weights/best.pt')

# 2. Run inference
# source can be a single image, a folder, or a video file
results = model.predict(
    source='test1.mp4',
    conf=0.25,      # Confidence threshold (0.0 to 1.0)
    save=True,      # Save the annotated image to 'runs/detect/predict'
    line_width=2,   # Set bounding box thickness
    show_labels=True,
    show_conf=True
)

# 3. Print summary of detections
for result in results:
    # 'boxes' contains detection data: coordinates, confidence, and class
    num_detections = len(result.boxes)
    print(f"Detected {num_detections} objects in {result.path}")

print("Done! Results saved in the 'runs/detect/predict' directory.")

