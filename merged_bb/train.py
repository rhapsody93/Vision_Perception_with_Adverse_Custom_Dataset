import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from ultralytics import YOLO
from multiprocessing import freeze_support

# dataset
dataset = [
    {
        "name": "MERGED",
        "data_yaml": "merged_dataset.yaml"
    }
]

# YOLO model & conditions setting
yolo_model = "yolov8l.pt"
epochs = 100
imgsz = 640
batch = 32
device = 0  # CUDA:0 (will use RTX 4060 GPU)

def main():
    for train_set in dataset:
        print(f"\nTraining Start: {train_set['name']}")

        try:
            model = YOLO(yolo_model)

            results = model.train(
                data=train_set['data_yaml'],
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                name=f"exp_{train_set['name']}",
                device=device
            )
            print(f"Train finished: {train_set['name']}\n")
        except Exception as e:
            print(f"Error has occurred: {train_set['name']}\nError: {e}\n")

if __name__ == '__main__':
    freeze_support()
    main()