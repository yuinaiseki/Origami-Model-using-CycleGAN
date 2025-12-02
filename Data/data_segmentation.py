import os
import cv2, sys
import numpy as np
from ultralytics import YOLO


ANIMALS_SPLIT_ROOT = "dataset/clean/split/animals"
ORIGAMI_SPLIT_ROOT = "dataset/clean/split/origami"

OUTPUT_DIR = "dataset/clean/split/segmented"

MODEL_PATH = "yolov8x-seg.pt"

for split in ["trainB", "valB", "testB"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

def find_images(path):
    imgs = []
    for r, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(os.path.join(r, f))
    return sorted(imgs)

def progress_bar(current, total, bar_length=40):
    fraction = current / total
    filled_len = int(bar_length * fraction)

    bar = "*" * filled_len + "-" * (bar_length - filled_len)
    percent = int(fraction * 100)

    sys.stdout.write(f"\r   [{bar}] {percent}% ({current}/{total})")
    sys.stdout.flush()

class SegEngine:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_mask(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        H, W = img.shape[:2]
        result = self.model(img_path, verbose=False)[0]

        if result.masks is None:
            return np.zeros((H, W), dtype=np.uint8)

        masks = result.masks.data.cpu().numpy()

        if masks.size == 0:
            return np.zeros((H, W), dtype=np.uint8)

        merged = np.max(masks, axis=0)
        return (merged > 0.5).astype(np.uint8)


def process_animals_split(split, engine):
    print(f"\nProcessing animals/{split} ...")

    input_split_path = os.path.join(ANIMALS_SPLIT_ROOT, split)
    output_split_path = os.path.join(OUTPUT_DIR, split + "B")

    class_names = sorted(os.listdir(input_split_path))
    ordered_classes = class_names

    for class_name in ordered_classes:
        class_input_dir = os.path.join(input_split_path, class_name)
        if not os.path.isdir(class_input_dir):
            continue

        print(f"\n Class: {class_name}")
        class_output_dir = os.path.join(output_split_path, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        images = find_images(class_input_dir)
        total_imgs = len(images)
        counter = 1

        for idx, img_path in enumerate(images, start=1):
            img = cv2.imread(img_path)
            if img is None:
                continue

            mask = engine.get_mask(img_path)

            new_name = f"{class_name}_{counter:04d}"

            out_img = os.path.join(class_output_dir, new_name + ".jpg")
            out_mask = os.path.join(class_output_dir, new_name + "_mask.png")

            cv2.imwrite(out_img, img)
            cv2.imwrite(out_mask, mask * 255)

            counter += 1

            progress_bar(idx, total_imgs)

        print("")  # newline after progress bar


def main():
    engine = SegEngine(MODEL_PATH)

    for split in ["val", "test"]:
        process_animals_split(split, engine)

    print("\nData segmentation completed for animals")


if __name__ == "__main__":
    main()