from pathlib import Path
import json
import random
import shutil
import open3d as o3d
import numpy as np

# 1. 경로 설정
pcd_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/autonomous_driving_adverse_weather/open_source_data/Training/01.raw_image/TS")
json_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/dataset/weather/autonomous_driving_adverse_weather/open_source_data/Training/02.labeling_data/TL/3D")
output_root = Path("C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/pcd/AIHub")

# 2. 유효한 PCD-JSON-IMAGE 쌍 수집
valid_triples = []
for scene_dir in pcd_root.iterdir():
    lidar_dir = scene_dir / "sensor_raw_data" / "lidar"
    cam_dir = scene_dir / "sensor_raw_data" / "camera"
    label_json_dir = json_root / scene_dir.name / "sensor_raw_data" / "camera"

    if not lidar_dir.exists() or not label_json_dir.exists() or not cam_dir.exists():
        continue

    for pcd_file in lidar_dir.glob("*.pcd"):
        base = pcd_file.stem
        label_file = label_json_dir / f"{base}.JSON"

        if not label_file.exists():
            continue

        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            jpg_name = json_data["information"]["filename"]
            img_path = cam_dir / jpg_name
            if not img_path.exists():
                continue
        except Exception as e:
            print(f"[ERROR parsing JSON] {label_file} - {e}")
            continue

        valid_triples.append((pcd_file, label_file, img_path))

print(f"총 유효한 (PCD, JSON, JPG) 쌍: {len(valid_triples)}")

# 3. 랜덤 추출 및 분할
random.seed(42)
random.shuffle(valid_triples)
sampled_triples = valid_triples[:10000]

n_total = len(sampled_triples)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.2)
n_test = n_total - n_train - n_val

split_dict = {
    "train": sampled_triples[:n_train],
    "val": sampled_triples[n_train:n_train + n_val],
    "test": sampled_triples[n_train + n_val:]
}

# 4. 변환 함수 정의
def pcd_to_bin(pcd_path, bin_path):
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    intensities = np.zeros((points.shape[0], 1), dtype=np.float32)
    points_intensity = np.hstack((points, intensities)).astype(np.float32)
    points_intensity.tofile(bin_path)

def json_to_kitti_txt(json_path, txt_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = []
    for ann in data.get("annotations", []):
        if ann.get("dimension") and ann.get("location") and ann.get("yaw") is not None:
            obj_type = ann.get("attribute", {}).get("type", "Unknown")
            h, w, l = ann["dimension"]
            x, y, z = ann["location"]
            ry = ann["yaw"]
            truncated = ann.get("attribute", {}).get("truncated", 0)
            occluded = ann.get("attribute", {}).get("occluded", 0)
            bbox = ann.get("bbox", [0, 0, 0, 0])
            line = f"{obj_type} {truncated} {occluded} 0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}"
            lines.append(line)

    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))

# 5. 저장 폴더 구조 생성 및 변환 수행
for split, triples in split_dict.items():
    img_split_dir = output_root / "images" / split
    lbl_split_dir = output_root / "labels" / split
    bin_split_dir = output_root / "PCDs" / split

    img_split_dir.mkdir(parents=True, exist_ok=True)
    lbl_split_dir.mkdir(parents=True, exist_ok=True)
    bin_split_dir.mkdir(parents=True, exist_ok=True)

    for i, (pcd_path, label_path, img_path) in enumerate(triples):
        base_name = f"{split}_{i:06d}"

        # 변환 저장
        pcd_to_bin(pcd_path, bin_split_dir / f"{base_name}.bin")
        json_to_kitti_txt(label_path, lbl_split_dir / f"{base_name}.txt")
        shutil.copy(img_path, img_split_dir / f"{base_name}.jpg")

    print(f"[{split.upper()}] Saved: {len(triples)}pairs")

# 6. 요약 출력
print("\nFinished")
print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")