import tensorflow as tf
from tensorflow.keras import layers
try:
    import tensorflow_addons as tfa
    NormLayer = tfa.layers.InstanceNormalization
except ImportError:
    # fallback to BatchNorm if addons not installed
    NormLayer = layers.BatchNormalization


# ===========================
# Residual Block
# ===========================
def residual_block(x, filters):
    y = layers.Conv2D(filters, 3, padding="same")(x)
    y = NormLayer()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same")(y)
    y = NormLayer()(y)
    return layers.add([x, y])


# ===========================
# Generator: ResNet-style (CycleGAN)
# ===========================
def build_generator(image_size=128, channels=3, n_blocks=6):
    inputs = layers.Input(shape=(image_size, image_size, channels))

    # Downsampling
    x = layers.Conv2D(64, 7, strides=1, padding="same")(inputs)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(n_blocks):
        x = residual_block(x, 256)

    # Upsampling (use resize + conv to avoid checkerboard artifacts)
    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    x = layers.Conv2D(128, 3, strides=1, padding="same")(x)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
    x = layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    # Output layer
    x = layers.Conv2D(channels, 7, strides=1, padding="same")(x)
    outputs = layers.Activation("tanh")(x)

    return tf.keras.Model(inputs, outputs, name="Generator")


# ===========================
# Discriminator: PatchGAN
# ===========================
def build_discriminator(image_size=128, channels=3):
    inputs = layers.Input(shape=(image_size, image_size, channels))

    x = layers.Conv2D(64, 4, strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = NormLayer()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = NormLayer()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(512, 4, strides=1, padding="same")(x)
    x = NormLayer()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Output PatchGAN map (no sigmoid for LSGAN)
    outputs = layers.Conv2D(1, 4, strides=1, padding="same")(x)

    return tf.keras.Model(inputs, outputs, name="Discriminator")
