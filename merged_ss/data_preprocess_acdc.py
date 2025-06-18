from pathlib import Path
import shutil
import random
from PIL import Image

# 입력 경로
original_image_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/DAWN_and_ACDC/ACDC/rgb_anon_trainvaltest/rgb_anon")
original_label_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/DAWN_and_ACDC/ACDC/gt_trainval/gt")

# 출력 경로
output_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_ss/ACDC")
(output_root / "images").mkdir(parents=True, exist_ok=True)
(output_root / "labels").mkdir(parents=True, exist_ok=True)

weather_types = ["fog", "night", "rain", "snow"]
splits = ["train", "val"]  # test split은 라벨이 없음

missing_images = []
missing_labels = []
matched_pairs = []

for weather in weather_types:
    for split in splits:
        rgb_img_root = original_image_root / weather / split
        label_root = original_label_root / weather / split

        if not rgb_img_root.exists() or not label_root.exists():
            print(f"No folder: {rgb_img_root} or {label_root}")
            continue

        for city_folder in rgb_img_root.iterdir():
            if not city_folder.is_dir():
                continue
            for rgb_img in city_folder.glob("*_rgb_anon.png"):
                image_id = rgb_img.stem.replace("_rgb_anon", "")
                label_img = label_root / city_folder.name / f"{image_id}_gt_labelTrainIds.png"

                if not label_img.exists():
                    missing_labels.append(str(label_img))
                    continue

                matched_pairs.append((rgb_img, label_img))

print(f"\nTotal matched image-label pairs: {len(matched_pairs)}")

# 데이터 셔플 및 저장
random.seed(42)
random.shuffle(matched_pairs)

for idx, (img_path, label_path) in enumerate(matched_pairs):
    base_name = f"{img_path.parent.name}_{img_path.stem}"
    new_img_name = f"{base_name}.jpg"
    new_lbl_name = f"{label_path.parent.name}_{label_path.name}"

    dst_img = output_root / "images" / new_img_name
    dst_lbl = output_root / "labels" / new_lbl_name

    # PNG 이미지 -> JPG 변환
    try:
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(dst_img, format="JPEG", quality=95)
    except Exception as e:
        print(f"[Error] image conversion failed: {img_path} - {e}")
        continue

    # shutil.copy(img_path, dst_img)
    shutil.copy(label_path, dst_lbl)

    print(f"Saved: {dst_img.name}, {dst_lbl.name}")

# 데이터 추출 결과 출력
print("\n데이터 추출 결과 요약:")
print(f"  - 누락된 이미지 수: {len(missing_images)}")
print(f"  - 누락된 라벨 수: {len(missing_labels)}")
print(f"  - 총 저장된 image-label pair 수: {len(matched_pairs)}")
