import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import random

def load_image(path, img_size=256):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img

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

def show_batch(dataset, n=4):
    imgs = next(iter(dataset))
    imgs = (imgs + 1.0) / 2.0
    plt.figure(figsize=(2*n, 2))
    for i in range(min(n, imgs.shape[0])):
        plt.subplot(1, n, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
    plt.show()
