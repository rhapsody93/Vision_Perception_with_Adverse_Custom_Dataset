from pathlib import Path
import json
import shutil
import random
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# 통합 클래스 매핑 정의
combined_class_mapping = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4, "pole": 5,
    "traffic light": 6, "traffic sign": 7, "vegetation": 8, "terrain": 9,
    "sky": 10, "person": 11, "rider": 12, "car": 13, "truck": 14,
    "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18,
    "road mark": 19, "lane": 20, "crosswalk": 21
}

# AIHub 클래스 원본 -> 통합 클래스 매핑
aihub_class_map = {
    "freespace": "road",
    "sideWalk": "sidewalk",
    "pedestrian": "person",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "schoolBus": "bus",
    "truck": "truck",
    "otherCar": "car",
    "vehicle": "car",
    "policeCar": "car",
    "ambulance": "car",
    "rider": "rider",
    "whiteLane": "lane",
    "yellowLane": "lane",
    "blueLane": "lane",
    "redLane": "lane",
    "stopLine": "lane",
    "roadMark": "road mark",
    "trafficSign": "traffic sign",
    "trafficLight": "traffic light",
    "fence": "fence",
    "crosswalk": "crosswalk"
}

# 경로 설정
aihub_image_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_ss/AIHub/images")  # JPG
aihub_label_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_ss/AIHub/labels")  # JSON
acdc_image_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_ss/ACDC/images")    # PNG
acdc_label_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_ss/ACDC/labels")    # PNG(labelTrainIds)

output_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_ss/MERGED")
merged_image_root = output_root / "images"
merged_label_root = output_root / "labels"
merged_image_root.mkdir(parents=True, exist_ok=True)
merged_label_root.mkdir(parents=True, exist_ok=True)


# 1. AIHub: JSON -> 마스크 PNG 변환
def convert_json_to_mask(json_path, resolution=(1920, 1080)):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mask = np.zeros(resolution[::-1], dtype=np.uint8)  # (H, W)
    for ann in data.get("annotations", []):
        poly = ann.get("polygon", [])
        label = ann.get("class", "")
        if label not in aihub_class_map:
            continue
        mapped_label = aihub_class_map[label]
        class_id = combined_class_mapping.get(mapped_label, 255)
        if class_id == 255 or len(poly) < 6:
            continue
        coords = np.array(poly, np.int32).reshape(-1, 2)
        Image.fromarray(mask).load()  # ensure alloc
        ImageDraw = Image.fromarray(mask)
        ImageDraw = ImageDraw.convert('L')
        ImageDraw = np.array(ImageDraw)
        cv2.fillPoly(mask, [coords], int(class_id))
    return mask

aihub_pairs = list(zip(sorted(aihub_image_root.glob("*.jpg")), sorted(aihub_label_root.glob("*.json"))))
aihub_sampled = random.sample(aihub_pairs, 7994)

print(f"Processing AIHub ({len(aihub_sampled)} samples)...")
for idx, (img_path, json_path) in enumerate(tqdm(aihub_sampled)):
    new_name = f"aihub_{idx:05d}"
    shutil.copy(img_path, merged_image_root / f"{new_name}.jpg")
    mask = convert_json_to_mask(json_path)
    Image.fromarray(mask).save(merged_label_root / f"{new_name}.png")


# 2. ACDC: PNG -> 통합 클래스 매핑 적용
# ACDC -> 통합 클래스 매핑 (labelTrainId 기준)
acdc_label_map = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation", 9: "terrain",
    10: "sky", 11: "person", 12: "rider", 13: "car", 14: "truck",
    15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"
}

def remap_acdc_mask(label_img_path):
    mask = np.array(Image.open(label_img_path))
    remapped_mask = np.full_like(mask, 255)
    for k, v in acdc_label_map.items():
        if v in combined_class_mapping:
            remapped_mask[mask == k] = combined_class_mapping[v]
    return remapped_mask

acdc_img_paths = sorted(acdc_image_root.glob("*.jpg"))
acdc_lbl_paths = sorted(acdc_label_root.glob("*.png"))

print(f"Processing ACDC ({len(acdc_img_paths)} samples)...")
for idx, (img_path, lbl_path) in enumerate(tqdm(zip(acdc_img_paths, acdc_lbl_paths), total=len(acdc_img_paths))):
    new_name = f"acdc_{idx:05d}"
    # 이미지 복사
    dst_img_path = merged_image_root / f"{new_name}.jpg"
    shutil.copy(img_path, dst_img_path)
    # 라벨: 마스크 변환
    remapped = remap_acdc_mask(lbl_path)
    Image.fromarray(remapped).save(merged_label_root / f"{new_name}.png")

# 3. train/val/test 분할
print("\nSplitting MERGED into train/val/test (7:2:1)")

#  rgb image-png mask image pair list 수집
all_pairs = list(zip(sorted(merged_image_root.glob("*.jpg")), sorted(merged_label_root.glob("*.png"))))
random.seed(42)
random.shuffle(all_pairs)

total = len(all_pairs)
train_ratio = int(total * 0.7)
val_ratio = int(total * 0.2)

split_dict = {
    "train": all_pairs[:train_ratio],
    "val": all_pairs[train_ratio:train_ratio + val_ratio],
    "test": all_pairs[train_ratio + val_ratio:]
}

# 디렉토리 생성 및 복사
for split in ["train", "val", "test"]:
    (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in tqdm(split_dict[split], desc=f"Saving {split}"):
        shutil.move(str(img_path), output_root / "images" / split / img_path.name)
        shutil.move(str(lbl_path), output_root / "labels" / split / lbl_path.name)

print("\nMERGED 데이터셋 생성 완료")
print(f"총 이미지 수: {len(list(merged_image_root.glob('*.jpg')))}")
print(f"총 레이블 수: {len(list(merged_label_root.glob('*.png')))}")
print(f"  - train: {len(split_dict['train'])}")
print(f"  - val:   {len(split_dict['val'])}")
print(f"  - test:  {len(split_dict['test'])}")

