import tensorflow as tf
from tensorflow.keras import layers

try:
    import tensorflow_addons as tfa
    NormLayer = tfa.layers.InstanceNormalization
except:
    NormLayer = layers.BatchNormalization


def residual_block(x, filters):
    y = layers.Conv2D(filters, 3, padding="same")(x)
    y = NormLayer()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same")(y)
    y = NormLayer()(y)
    return layers.add([x, y])


def build_generator(image_size=256, in_channels=4, out_channels=3, n_blocks=9):
    inputs = layers.Input(shape=(image_size, image_size, in_channels))

    x = layers.Conv2D(64, 7, strides=1, padding="same")(inputs)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    for _ in range(n_blocks):
        x = residual_block(x, 256)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = NormLayer()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(out_channels, 7, padding="same")(x)
    outputs = layers.Activation("tanh")(x)

    return tf.keras.Model(inputs, outputs, name="Generator")


def build_discriminator(image_size=256, channels=3):
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

    outputs = layers.Conv2D(1, 4, strides=1, padding="same")(x)

    return tf.keras.Model(inputs, outputs, name="Discriminator")
