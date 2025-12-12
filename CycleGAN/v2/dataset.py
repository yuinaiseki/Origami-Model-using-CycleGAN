import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import random

def load_image(path, img_size=256):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size])
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img

def load_mask(mask_path, img_size=256):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(mask, [img_size, img_size])
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask

def get_dataset(folder_pattern, batch_size=4, img_size=256):
    files = glob.glob(folder_pattern)
    if len(files) == 0:
        raise ValueError(f"No images found for pattern: {folder_pattern}")
    random.shuffle(files)
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(buffer_size=max(100, len(files)))
    ds = ds.map(lambda x: load_image(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def get_dataset_with_mask(image_pattern, mask_pattern, batch_size=4, img_size=256):
    image_files = sorted(glob.glob(image_pattern))
    mask_files  = sorted(glob.glob(mask_pattern))

    if len(image_files) == 0:
        raise ValueError(f"No images found for pattern: {image_pattern}")
    if len(mask_files) == 0:
        raise ValueError(f"No mask images found for pattern: {mask_pattern}")

    if len(image_files) != len(mask_files):
        print("Warning: image and mask counts differ. Using sorted order pairing.")

    paired = list(zip(image_files, mask_files))
    random.shuffle(paired)

    ds = tf.data.Dataset.from_tensor_slices(paired)

    def _load_pair(img_path, mask_path):
        img = load_image(img_path, img_size)
        msk = load_mask(mask_path, img_size)
        return tf.concat([img, msk], axis=-1)

    ds = ds.map(lambda a, b: _load_pair(a, b), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=max(100, len(paired)))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def show_batch(dataset, n=4):
    imgs = next(iter(dataset))
    imgs = (imgs + 1.0) / 2.0
    plt.figure(figsize=(2*n, 2))
    for i in range(min(n, imgs.shape[0])):
        plt.subplot(1, n, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
    plt.show()