from pathlib import Path
import json
import random
import shutil
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def create_kitti_calib_file(meta_json_path, calib_txt_path):
    if not meta_json_path.exists():
        print(f"[WARNING] Meta calibration file not found: {meta_json_path}")
        return

    with open(meta_json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    try:
        # Intrinsics
        camera_key = 'front'
        cam_intr = meta["calibration"]["camera"][camera_key]["Intrinsic"]
        fx, fy = cam_intr["Fx"], cam_intr["Fy"]
        cx, cy = cam_intr["Cx"], cam_intr["Cy"]
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        P2 = np.zeros((3, 4))
        P2[:3, :3] = intrinsics  # KITTI P2 matrix

        # 2. Extrinsics
        cam_ext = meta["calibration"]["camera"][camera_key]["Extrinsic"]
        tx, ty, tz = cam_ext["Tx"], cam_ext["Ty"], cam_ext["Tz"]
        rx, ry, rz = cam_ext["Rx"], cam_ext["Ry"], cam_ext["Rz"]  # in degrees

        # Convert Euler angles to rotation matrix (assume 'xyz' order)
        R_cam = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_cam
        T[:3, 3] = [tx, ty, tz]

        Tr_velo_to_cam = T[:3, :]  # shape (3, 4)

        # 3. KITTI rectification matrix (identity)
        R0_rect = np.eye(3)

        with open(calib_txt_path, 'w') as f:
            f.write("P0: " + " ".join(["0"] * 12) + "\n")
            f.write("P1: " + " ".join(["0"] * 12) + "\n")
            f.write("P2: " + " ".join([f"{x:.12e}" for x in P2.flatten()]) + "\n")
            f.write("P3: " + " ".join(["0"] * 12) + "\n")
            f.write("R0_rect: " + " ".join([f"{x:.12e}" for x in R0_rect.flatten()]) + "\n")
            f.write("Tr_velo_to_cam: " + " ".join([f"{x:.12e}" for x in Tr_velo_to_cam.flatten()]) + "\n")

    except Exception as e:
        print(f"[ERROR parsing meta] {meta_json_path} - {e}")

# 5. 데이터셋 구조(KITTI format)
image_sets_dir = output_root / "ImageSets"
image_sets_dir.mkdir(parents=True, exist_ok=True)

training_img_dir = output_root / "training" / "image_2"
training_bin_dir = output_root / "training" / "velodyne"
training_label_dir = output_root / "training" / "label_2"
training_calib_dir = output_root / "training" / "calib"

training_img_dir.mkdir(parents=True, exist_ok=True)
training_bin_dir.mkdir(parents=True, exist_ok=True)
training_label_dir.mkdir(parents=True, exist_ok=True)
training_calib_dir.mkdir(parents=True, exist_ok=True)

# 6. train/val/test 별 파일 저장 및 txt 생성
index_counter = 0
split_txt_lines = {"train": [], "val": [], "test": []}

for split, triples in split_dict.items():
    for (pcd_path, label_path, img_path) in triples:
        base_name = f"{index_counter:06d}"
        split_txt_lines[split].append(base_name)

        # 저장 (모두 training/ 폴더 아래)
        pcd_to_bin(pcd_path, training_bin_dir / f"{base_name}.bin")
        shutil.copy(img_path, training_img_dir / f"{base_name}.jpg")
        json_to_kitti_txt(label_path, training_label_dir / f"{base_name}.txt")

        # calibration file conversion
        scene_folder = pcd_path.parents[2].name  # ex: "08_084144_221012"
        meta_path = pcd_root / scene_folder / f"{scene_folder}_meta_data.json"
        calib_out = training_calib_dir / f"{base_name}.txt"
        create_kitti_calib_file(meta_path, calib_out)

        index_counter += 1

# 7. ImageSets/*.txt(train, val, test) 파일 작성
for split in ["train", "val", "test"]:
    split_txt_path = image_sets_dir / f"{split}.txt"
    with open(split_txt_path, 'w') as f:
        f.write('\n'.join(split_txt_lines[split]))


# 8. 결과
print("\nFinished")
print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")