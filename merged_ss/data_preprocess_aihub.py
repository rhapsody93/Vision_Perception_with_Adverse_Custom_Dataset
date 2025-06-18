import os
import json
import shutil
import random
from pathlib import Path

# 경로 설정
original_image_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/autonomous_driving_adverse_weather/open_source_data/Training/01.raw_image/TS")
original_label_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/autonomous_driving_adverse_weather/open_source_data/Training/02.labeling_data/TL/2D")
output_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_ss/AIHub")

# 출력 디렉토리 생성
(output_root / "images").mkdir(parents=True, exist_ok=True)
(output_root / "labels").mkdir(parents=True, exist_ok=True)

# 모든 유효 JSON 파일 경로 수집
json_files = list(original_label_root.glob("**/sensor_raw_data/camera/*.json"))
print(f"Total number of JSON files: {len(json_files)}")

valid_pairs = []

for json_path in json_files:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # polygon 존재 여부 확인
        if "annotations" not in data or len(data["annotations"]) == 0:
            continue

        has_valid_polygon = any(
            ann.get("polygon") and isinstance(ann["polygon"], list) and len(ann["polygon"]) >= 6
            for ann in data["annotations"]
        )
        if not has_valid_polygon:
            continue

        img_name = data["information"]["filename"]
        folder_id = json_path.parts[-4]  # ex. 08_084144_221012

        img_path = original_image_root / folder_id / "sensor_raw_data" / "camera" / img_name

        if img_path.exists():
            valid_pairs.append((img_path, json_path))
    except Exception as e:
        print(f"error has occurred: {json_path} - {e}")
        continue

print(f"The number of valid image-label pairs: {len(valid_pairs)}")

# 7,994쌍 랜덤 추출
random.seed(42)
random.shuffle(valid_pairs)
sampled_pairs = valid_pairs[:7994]

# 복사 수행
for idx, (img_path, label_path) in enumerate(sampled_pairs):
    new_img_name = f"img_{idx:05d}.jpg"
    new_lbl_name = f"img_{idx:05d}.json"

    shutil.copy(img_path, output_root / "images" / new_img_name)
    shutil.copy(label_path, output_root / "labels" / new_lbl_name)

print(f"image-label copy has finished : Total {len(sampled_pairs)} pairs")
