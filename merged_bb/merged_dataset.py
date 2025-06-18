import random
import shutil
from pathlib import Path
from PIL import Image

# DAWN, ACDC, AIHub의 전처리 결과 루트
dawn_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/DAWN")
acdc_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/ACDC")
aihub_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/AIHub")
output_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb")

# PNG → JPG 변환 함수 (ACDC만 적용)
def convert_png_to_jpg(root_dir):
    image_dir = root_dir / "images"
    for png_path in image_dir.glob("*.png"):
        jpg_path = png_path.with_suffix(".jpg")
        try:
            with Image.open(png_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(jpg_path, "JPEG", quality=95)
            png_path.unlink()  # 변환 후 원본 PNG 삭제
            print(f"[변환 완료] {png_path.name} → {jpg_path.name}")
        except Exception as e:
            print(f"[오류] {png_path.name} 변환 실패: {e}")

# ACDC 이미지 확장자 변환 수행(png → jpg)
convert_png_to_jpg(acdc_root)

# 이미지 및 라벨 쌍 수집 함수 (jpg + png 지원)
def collect_data(root_dir):
    data_pairs = []
    image_dir = root_dir / "images"
    label_dir = root_dir / "labels"
    if not image_dir.exists() or not label_dir.exists():
        print(f"[경고] {root_dir.name} - images/labels 폴더가 존재하지 않음. 건너뜀.")
        return data_pairs

    for ext in ["*.jpg", "*.png"]:
        for img_file in image_dir.glob(ext):
            label_file = label_dir / (img_file.stem + ".txt")
            if label_file.exists():
                label_lines = label_file.read_text().strip().splitlines()
                if label_lines:
                    data_pairs.append((img_file, label_lines))
    print(f"{root_dir.name} 데이터 수집 완료: 총 {len(data_pairs)}쌍")
    return data_pairs

# DAWN, ACDC, AIHub 데이터 수집
dawn_data = collect_data(dawn_root)
acdc_data = collect_data(acdc_root)
aihub_data = collect_data(aihub_root)

# 데이터 통합 및 셔플
combined_data = dawn_data + acdc_data + aihub_data
random.seed(42)
random.shuffle(combined_data)

# 7:2:1 비율로 데이터 분할
total = len(combined_data)
split1 = int(total * 0.7)
split2 = int(total * 0.9)
train_data = combined_data[:split1]
val_data = combined_data[split1:split2]
test_data = combined_data[split2:]

# 저장 함수
def save_split(data_split, split_name):
    image_out_dir = output_root / "MERGED" / "images" / split_name
    label_out_dir = output_root / "MERGED" / "labels" / split_name
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label_lines in data_split:
        new_img_path = image_out_dir / img_path.name
        new_lbl_path = label_out_dir / (img_path.stem + ".txt")
        shutil.copy(img_path, new_img_path)
        new_lbl_path.write_text("\n".join(label_lines))

# 저장 수행
save_split(train_data, "train")
save_split(val_data, "val")
save_split(test_data, "test")

# 통계 출력
print(f"총 통합 이미지 수: {total}")
print(f"  - Train: {len(train_data)} ({len(train_data)/total:.2%})")
print(f"  - Val:   {len(val_data)} ({len(val_data)/total:.2%})")
print(f"  - Test:  {len(test_data)} ({len(test_data)/total:.2%})")
