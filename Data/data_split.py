import os
import random
import shutil
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

MOUNT_ROOT = Path("dataset/clean")
ANIMALS_ROOT = MOUNT_ROOT / "animals_balanced"
ORIGAMI_ROOT = MOUNT_ROOT / "origami_images"
SPLIT_ROOT = MOUNT_ROOT / "split"

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def split_paths(paths, split_ratio):
    random.shuffle(paths)
    n = len(paths)

    n_train = int(n * split_ratio["train"])
    n_val = int(n * split_ratio["val"])
    n_test = n - n_train - n_val

    return {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:]
    }


def copy_split(imgs_by_split, domain, classname):
    for split_name, img_paths in imgs_by_split.items():
        out_dir = SPLIT_ROOT / domain / split_name / classname
        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(img_paths)
        copied = 0
        skipped = 0

        print(f"\n[{domain}/{classname}] -> {split_name} ({total} files)")

        for i, src in enumerate(img_paths, start=1):
            dst = out_dir / src.name

            if dst.exists():
                skipped += 1
                continue

            shutil.copy2(src, dst)
            copied += 1

            if i % 50 == 0 or i == total:
                print(f"  {split_name}: {i}/{total} processed "
                      f"(copied={copied}, skipped={skipped})", end="\r")

        print(f"\n  Done {split_name}: copied={copied}, skipped={skipped}")


def process_domain(root: Path, domain_name: str):
    class_dirs = [d for d in root.iterdir() if d.is_dir()]

    print(f"\nProcessing domain '{domain_name}' with {len(class_dirs)} classes")

    for class_dir in class_dirs:
        classname = class_dir.name
        img_paths = [p for p in class_dir.iterdir() if is_image(p)]

        if len(img_paths) == 0:
            print(f"  [SKIP] No images in {domain_name}/{classname}")
            continue

        print(f"\n=== Class: {domain_name}/{classname} ({len(img_paths)} images) ===")
        imgs_by_split = split_paths(img_paths, SPLITS)
        copy_split(imgs_by_split, domain_name, classname)


def main():
    random.seed(42)
    SPLIT_ROOT.mkdir(exist_ok=True)

    process_domain(ANIMALS_ROOT, "animals")
    process_domain(ORIGAMI_ROOT, "origami")

    print("\nSplit completed at:", SPLIT_ROOT)


if __name__ == "__main__":
    main()
