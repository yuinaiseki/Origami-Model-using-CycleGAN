import os
import glob
import tensorflow as tf

DATA_ROOT = "/home/ubuntu/projects/data"
ORIGAMI_INPUT = os.path.join(DATA_ROOT, "origami_images")
AUG_OUTPUT = os.path.join(DATA_ROOT, "augmented_data", "origami_images")
N_AUG_PER_IMAGE = 2
IMG_SIZE = 1024
JITTER_SIZE = 1152

def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img

def random_jitter(img, img_size=IMG_SIZE, jitter_size=JITTER_SIZE):
    img = tf.image.resize(img, [jitter_size, jitter_size])
    img = tf.image.random_crop(img, [img_size, img_size, 3])
    img = tf.image.random_flip_left_right(img)
    return img

def color_augment(img):
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    img = tf.image.random_hue(img, max_delta=0.02)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img

def augment_once(img):
    img = random_jitter(img)
    img = color_augment(img)
    return tf.clip_by_value(img, 0.0, 1.0)

def save_image(img, path):
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    encoded = tf.io.encode_jpeg(img, quality=95)
    tf.io.write_file(path, encoded)


def augment_origami():
    os.makedirs(AUG_OUTPUT, exist_ok=True)

    patterns = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(ORIGAMI_INPUT, p)))

    print(f"[INFO] Found {len(files)} origami images.")
    print(f"[INFO] Saving augmented versions to: {AUG_OUTPUT}")

    for idx, path in enumerate(files):
        try:
            base = os.path.splitext(os.path.basename(path))[0]
            img = decode_image(path)

            for k in range(N_AUG_PER_IMAGE):
                aug = augment_once(img)
                out_name = f"{base}_aug{k+1}.jpg"
                out_path = os.path.join(AUG_OUTPUT, out_name)
                save_image(aug, out_path)

            if idx % 50 == 0:
                print(f"[INFO] Processed {idx}/{len(files)} images")

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    print("[DONE] Augmentation complete!")

if __name__ == "__main__":
    augment_origami()
