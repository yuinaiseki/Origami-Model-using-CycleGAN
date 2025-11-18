import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import random

AUTOTUNE = tf.data.AUTOTUNE

def _decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img

def _random_jitter(img, img_size=256, jitter_size=286):
    img = tf.image.resize(img, [jitter_size, jitter_size])
    img = tf.image.random_crop(img, [img_size, img_size, 3])
    img = tf.image.random_flip_left_right(img)
    return img

def _color_augment(img):
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    img = tf.image.random_hue(img, max_delta=0.02)
    # small gaussian noise
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img

# def load_image(path, img_size=256):
#     img = tf.io.read_file(path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, [img_size, img_size])
#     img = (tf.cast(img, tf.float32) / 127.5) - 1.0
#     return img

def load_image(path, img_size=256, augment=True, jitter_size=286):
    img = _decode_image(path) 
    if augment:
        img = _random_jitter(img, img_size=img_size, jitter_size=jitter_size)
        img = _color_augment(img)
    else:
        #plain deterministic resize for val/test
        img = tf.image.resize(img, [img_size, img_size])
    img = (img * 2.0) - 1.0
    return img

# def get_dataset(folder_pattern, batch_size=4, img_size=256):
#     files = glob.glob(folder_pattern)
#     if len(files) == 0:
#         raise ValueError(f"No images found for pattern: {folder_pattern}")
#     random.shuffle(files)
#     ds = tf.data.Dataset.from_tensor_slices(files)
#     ds = ds.shuffle(buffer_size=max(100, len(files)))
#     ds = ds.map(lambda x: load_image(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
#     ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     return ds
def get_dataset(folder_pattern, batch_size=4, img_size=256, augment=True,jitter_size=286, shuffle=True, repeat=False):
    files = glob.glob(folder_pattern)
    if len(files) == 0:
        raise ValueError(f"No images found for pattern: {folder_pattern}")
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(100, len(files)), reshuffle_each_iteration=True)
    ds = ds.map(lambda x: load_image(x, img_size=img_size, augment=augment, jitter_size=jitter_size), num_parallel_calls=AUTOTUNE,)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def show_batch(dataset, n=4):
    imgs = next(iter(dataset))
    imgs = (imgs + 1.0) / 2.0  
    plt.figure(figsize=(2 * n, 2))
    for i in range(min(n, imgs.shape[0])):
        plt.subplot(1, n, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
    plt.show()
