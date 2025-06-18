import os
from pathlib import Path
import shutil
import random

# DAWN 원본 데이터 경로
original_image_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/DAWN_and_ACDC/DAWN")

# 출력 경로
output_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb")

weather_types = ["Fog", "Rain", "Sand", "Snow"]

# 클래스 재매핑(DAWN 기준 → YOLO 클래스 ID)
class_map = {
    1: 0,  # 1-person(original txt class ID)
    2: 1,  # 2-bicycle(original txt class ID)
    3: 2,  # 3-car(original txt class ID)
    4: 3,  # 4-motorcycle(original txt class ID)
    6: 5,  # 6-bus(original txt class ID)
    8: 7   # 8-truck(original txt class ID)
}

all_data = []

for weather in weather_types:
    label_dir = original_image_root / weather / weather / f"{weather}_YOLO_darknet"
    image_dir = original_image_root / weather / weather

    if not label_dir.exists():
        print(f"No label directory: {label_dir}")
        continue

    for label_file in label_dir.glob("*.txt"):
        print(f"Processing label file: {label_file}")  # 레이블 파일 출력
        lines = label_file.read_text().strip().splitlines()
        out_lines = []
        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            if cls in class_map:
                parts[0] = str(class_map[cls])
                out_lines.append(" ".join(parts))

        image_name = label_file.with_suffix(".jpg").name
        image_path = image_dir / image_name
        print(f"Checking image path: {image_path}")  # 이미지 경로 출력

        if image_path.exists() and out_lines:
            all_data.append((image_path, label_file.name, out_lines))
        else:
            print(f"No image or No label: {image_path}")

# 저장 경로 생성
image_out_dir = output_root / "DAWN" / "images"
label_out_dir = output_root / "DAWN" / "labels"
image_out_dir.mkdir(parents=True, exist_ok=True)
label_out_dir.mkdir(parents=True, exist_ok=True)

# 저장
for idx, (image_path, label_name, out_lines) in enumerate(all_data):
    image_out = image_out_dir / image_path.name
    label_out = label_out_dir / label_name

    shutil.copy(image_path, image_out)
    label_out.write_text("\n".join(out_lines))

print(f"총 이미지 수: {len(all_data)}개가 images/labels 폴더에 저장되었습니다.")