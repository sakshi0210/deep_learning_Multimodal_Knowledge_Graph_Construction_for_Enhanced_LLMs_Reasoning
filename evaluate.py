import model_code         # your SE-C2f patch is applied here
from ultralytics import YOLO

# Load modified YOLOv8 (SE injected into C2f)
model = YOLO("yolov8s.pt")

# Evaluate on COCO test set
metrics = model.val(
    data="coco_test.yaml",
    split="test",
    imgsz=640,
    save_json=True,    # generates COCO-style evaluation json
    save_txt=True,     # saves prediction labels
       
)

# Print all metrics
print("\n===== YOLOv8 + SE-C2f Accuracy Metrics =====")
print(f"mAP@50:       {metrics.box.map50:.4f}")
print(f"mAP@50-95:    {metrics.box.map:.4f}")
print(f"Precision:    {metrics.box.mp:.4f}")
print(f"Recall:       {metrics.box.mr:.4f}")
print(f"F1-score:     {(2 * metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr):.4f}")

# Per-class AP
print("\n===== Per-class Average Precision (AP) =====")
for i, ap in enumerate(metrics.box.maps):
    print(f"Class {i:02d}: {ap:.4f}")
