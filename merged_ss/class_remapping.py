import os
from pathlib import Path


def remap_classes(labels_path):
    # 9번 클래스를 4번으로 매핑
    for label_file in Path(labels_path).glob('*.txt'):
        modified_lines = []
        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                # 9번 클래스를 4번으로 변경
                if class_id == 9:
                    parts[0] = '4'
                modified_lines.append(' '.join(parts) + '\n')

        with open(label_file, 'w') as f:
            f.writelines(modified_lines)


# 경로 설정
train_path = "C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/MERGED/labels/train"
val_path = "C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/MERGED/labels/val"
test_path = "C:/Users/USER/anaconda3/envs/capstone_pt/capstone_project_03/merged_bb/MERGED/labels/test"


print("Remapping training labels...")
remap_classes(train_path)
print("\nRemapping validation labels...")
remap_classes(val_path)
print("\nRemapping test labels...")
remap_classes(test_path)
