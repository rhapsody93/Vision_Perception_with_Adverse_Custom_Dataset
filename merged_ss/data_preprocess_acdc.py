import json
from pathlib import Path
import shutil
import random
from collections import defaultdict

# ACDC 원본 데이터 경로
original_image_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/DAWN_and_ACDC/ACDC/rgb_anon_trainvaltest/rgb_anon")
original_label_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/DAWN_and_ACDC/ACDC/gt_detection_trainval/gt_detection")

# 출력 경로
output_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/ACDC")

weather_types = ["fog", "night", "rain", "snow"]

# 클래스 재매핑(ACDC 기준 → YOLO 클래스 ID)
class_map = {
    "person": 0,    # 24-person(original JSON category ID)
    "rider": 0,     # 25-rider(original JSON category ID)
    "bicycle": 1,   # 33-bicycle(original JSON category ID)
    "car": 2,       # 26-car(original JSON category ID)
    "motorcycle": 3,# 32-motorcycle(original JSON category ID)
    "bus": 5,       # 28-bus(original JSON category ID)
    "train": 6,     # 31-train(original JSON category ID)
    "truck": 7      # 27-truck(original JSON category ID)
}

data_pairs = []
excluded_images = []
missing_images = []

for weather in weather_types:
    for split in ["train", "val", "test"]:
        if split == "test":
            json_path = original_label_root / weather / f"instancesonly_{weather}_{split}_image_info.json"
        else:
            json_path = original_label_root / weather / f"instancesonly_{weather}_{split}_gt_detection.json"

        if not json_path.exists():
            print(f"JSON file missed: {json_path}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        annotations = {}

        for ann in data.get("annotations", []):
            cat_id = ann['category_id']
            cat_name = next((c['name'] for c in data['categories'] if c['id'] == cat_id), None)
            if cat_name not in class_map:
                continue

            image_id = ann['image_id']
            filename = id_to_filename[image_id]
            bbox = ann['bbox']
            xc = (bbox[0] + bbox[2] / 2) / 1920
            yc = (bbox[1] + bbox[3] / 2) / 1080
            w = bbox[2] / 1920
            h = bbox[3] / 1080
            line = f"{class_map[cat_name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

            if filename not in annotations:
                annotations[filename] = []
            annotations[filename].append(line)

        for img_id, filename in id_to_filename.items():
            src_img = original_image_root / filename
            if not src_img.exists():
                print(f"Image file missed: {src_img}")
                missing_images.append(str(src_img))
                continue

            label_lines = annotations.get(filename, [])
            if len(label_lines) == 0:
                print(f"No label → Image has been excluded: {filename}")
                excluded_images.append(str(src_img))
                continue

            data_pairs.append((src_img, label_lines, weather, split))

# 데이터 셔플 및 분할
random.shuffle(data_pairs)
total = len(data_pairs)
split1 = int(total * 0.7)
split2 = int(total * 0.9)

# ACDC 폴더 내 디렉토리 생성
(output_root / "images").mkdir(parents=True, exist_ok=True)
(output_root / "labels").mkdir(parents=True, exist_ok=True)

# 실제 복사 및 저장된 수량 카운트
successfully_saved = 0

# 파일 복사 및 라벨 저장
for idx, (img_path, lines, weather, _) in enumerate(data_pairs):
    # 고유 파일명 생성 (중복 방지)
    new_img_name = f"{weather}_{idx}_{img_path.name}"
    new_label_name = f"{weather}_{idx}_{img_path.stem}.txt"

    img_out_path = output_root / "images" / new_img_name
    label_out_path = output_root / "labels" / new_label_name

    try:
        shutil.copy(img_path, img_out_path)
        label_out_path.write_text("\n".join(lines))
        successfully_saved += 1
        print(f"Saved images: {new_img_name}, Saved labels: {new_label_name}")
    except Exception as e:
        print(f"Failed to save {img_path.name}: {e}")

# 최종 통계 출력
print("\nPreprocessing Summary:")
print(f"  - 처리된 최종 이미지 수: {len(data_pairs)}")
print(f"  - 실제 저장된 이미지 수: {successfully_saved}")
print(f"  - 레이블 없는 이미지 제외 수: {len(excluded_images)}")
print(f"  - 실제 이미지 파일 누락 수: {len(missing_images)}")
print(f"  - 총 JSON 등록 이미지 수: {len(data_pairs) + len(excluded_images) + len(missing_images)}")
