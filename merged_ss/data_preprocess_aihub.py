import os
import json
import shutil
import random
from glob import glob
from pathlib import Path

# 경로 설정
original_image_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/autonomous_driving_adverse_weather/open_source_data/Training/01.raw_image/TS")
original_label_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/autonomous_driving_adverse_weather/open_source_data/Training/02.labeling_data/TL/2D")
output_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/AIHub")

# 클래스 매핑 (COCO 형식)
class_map = {
    "pedestrian": 0,    # person
    "rider": 0,         # person
    "bicycle": 1,       # bicycle
    "vehicle": 2,       # car
    "otherCar": 2,      # car
    "ambulance": 2,     # car
    "policeCar": 2,     # car
    "motorcycle": 3,    # motorcycle
    "bus": 5,           # bus
    "schoolBus": 5,     # bus
    "truck": 7,         # truck
    "trafficLight": 9   # traffic light
}


# AIHub = image_paths[:5823]

# polygon → YOLO bbox 변환
def convert_annotation(json_path, image_w, image_h):
    yolo_lines = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "annotations" not in data or not data["annotations"]:
            print(f" No annotations: {json_path}")
            return []

        for ann in data["annotations"]:
            label = ann.get("class")
            if label not in class_map:
                print(f"filtered class: {label}")
                continue

            class_id = class_map[label]
            polygon = ann.get("polygon", [])
            if not polygon or len(polygon) < 6:
                print(f"polygon length error: {polygon}")
                continue

            x_coords = polygon[0::2]
            y_coords = polygon[1::2]
            x_min = max(min(x_coords), 0)
            x_max = min(max(x_coords), image_w)
            y_min = max(min(y_coords), 0)
            y_max = min(max(y_coords), image_h)
            x_center = (x_min + x_max) / 2 / image_w
            y_center = (y_min + y_max) / 2 / image_h
            width = (x_max - x_min) / image_w
            height = (y_max - y_min) / image_h

            if width <= 0 or height <= 0:
                print(f"invalid bbox (0 or negative size): {label}, {polygon}")
                continue

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    except Exception as e:
        print(f"JSON parsing failed: {json_path}, error: {e}")
        return []

    return yolo_lines

# JSON 라벨 경로 추출
def find_json_path(image_path):
    image_path = Path(image_path)
    ts_id = image_path.parents[2].name
    filename = image_path.stem + ".json"
    json_path = os.path.join(original_label_root, ts_id, "sensor_raw_data", "camera", filename)
    return json_path

image_out_dir = os.path.join(output_root, "images")
label_out_dir = os.path.join(output_root, "labels")
os.makedirs(image_out_dir, exist_ok=True)
os.makedirs(label_out_dir, exist_ok=True)

# 이미지 수집 및 분할
image_paths = glob(os.path.join(original_image_root, "*", "sensor_raw_data", "camera", "*.jpg"))
random.shuffle(image_paths)

valid_pairs = []
for img_path in image_paths:
    json_path = find_json_path(img_path)
    if not os.path.exists(json_path):
        continue

    yolo_lines = convert_annotation(json_path, 1920, 1080)
    if yolo_lines:
        valid_pairs.append((img_path, json_path, yolo_lines))
    if len(valid_pairs) >= 5823:
        break

# 이미지 및 라벨 복사/변환 처리
for img_path, json_path, yolo_lines in valid_pairs:
    img_name = os.path.basename(img_path)
    dst_img_path = os.path.join(image_out_dir, img_name)
    dst_label_path = os.path.join(label_out_dir, img_name.replace('.jpg', '.txt'))

    shutil.copy(img_path, dst_img_path)
    with open(dst_label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
    print(f"label creation finished: {dst_label_path} ({len(yolo_lines)}개 객체)")


# 통계 출력
print(f"\n최종 image-txt pair 수: {len(valid_pairs)}")
