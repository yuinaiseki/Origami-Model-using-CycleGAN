import tensorflow as tf
from pathlib import Path

MOUNT_ROOT = Path("../../data")
TRAIN_ORIGAMI = MOUNT_ROOT / "split" / "origami" / "train"

IMG_SIZE = 1024
JITTER_SIZE = 1152
N_AUG_PER_IMAGE = 4

IMG_EXTS = (".jpg", ".jpeg", ".png")


def is_image(path: Path):
    return path.suffix.lower() in IMG_EXTS


def decode_image(path: Path):
    img_bytes = tf.io.read_file(str(path))
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def random_jitter(img):
    img = tf.image.resize(img, [JITTER_SIZE, JITTER_SIZE])
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])
    img = tf.image.random_flip_left_right(img)
    return img


def color_augment(img):
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.3)
    return img


def augment_and_save(src: Path):
    img = decode_image(src)

    for i in range(1, N_AUG_PER_IMAGE + 1):
        aug = random_jitter(img)
        aug = color_augment(aug)
        aug = tf.clip_by_value(aug, 0, 1)

        aug_uint8 = tf.image.convert_image_dtype(aug, tf.uint8)
        encoded = tf.io.encode_png(aug_uint8)

        out_path = src.parent / f"{src.stem}_aug{i}.png"
        tf.io.write_file(str(out_path), encoded)


def main():
    class_dirs = [d for d in TRAIN_ORIGAMI.iterdir() if d.is_dir()]

    print("Augmenting origami data…")

    for class_dir in class_dirs:
        imgs = [p for p in class_dir.iterdir() if is_image(p)]

        print(f" → {class_dir.name}: {len(imgs)} original images")

        for idx, img in enumerate(imgs, 1):
            augment_and_save(img)
            if idx % 50 == 0:
                print(f"   {idx}/{len(imgs)} done")

    print("Augmentation complete.")


if __name__ == "__main__":
    main()
