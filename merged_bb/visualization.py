import cv2
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 클래스 이름 정의
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'traffic light', 'bus', 'train', 'truck']

# path 설정
merged_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/MERGED")
test_image_dir = merged_root / "images/test"
test_label_dir = merged_root / "labels/test"

# 저장할 시각화 결과 폴더
output_dir = Path("visual_results")
output_dir.mkdir(exist_ok=True)

# 예측 수행 경로 초기화
if Path("runs/detect/predict").exists():
    shutil.rmtree("runs/detect/predict")  # 이전 결과 제거

# best.pt 모델 로드 및 test set prediction
best_model_path = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/runs/detect/exp_MERGED/weights/best.pt")
model = YOLO(str(best_model_path))
model.predict(
    source=str(test_image_dir),
    save_txt=True,
    save_conf=True,
    save=False,
    imgsz=640,
    conf=0.25
)

# 경로 재정의: 예측 결과 라벨
predict_label_dir = Path("runs/detect/predict/labels")

def load_yolo_labels(label_path, img_width, img_height):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls, x, y, w, h = map(float, parts[:5])
            x1 = int((x - w / 2) * img_width)
            y1 = int((y - h / 2) * img_height)
            x2 = int((x + w / 2) * img_width)
            y2 = int((y + h / 2) * img_height)
            boxes.append((int(cls), x1, y1, x2, y2))
    return boxes

def draw_boxes(image, boxes, color, label_prefix='GT'):
    for cls, x1, y1, x2, y2 in boxes:
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        label = f'{label_prefix}:{name}'
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def visualize_and_save(image_path):
    image_id = Path(image_path).stem
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    gt_label_path = test_label_dir / f"{image_id}.txt"
    pred_label_path = predict_label_dir / f"{image_id}.txt"

    gt_boxes = load_yolo_labels(gt_label_path, w, h)
    pred_boxes = load_yolo_labels(pred_label_path, w, h)

    draw_boxes(img, gt_boxes, color=(0, 255, 0), label_prefix='GT')       # Green for Ground Truth
    draw_boxes(img, pred_boxes, color=(255, 0, 0), label_prefix='Pred')   # Blue for Prediction

    save_path = output_dir / f"{image_id}_gt_vs_pred.jpg"
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img_bgr)

# 모든 test image에 대해 시각화 수행
image_list = sorted(list(test_image_dir.glob("*.jpg")))
print(f"Total test images: {len(image_list)}")

for img_path in image_list:
    visualize_and_save(img_path)

print(f"\nAll visualization result has been saved. → {output_dir.resolve()}")
