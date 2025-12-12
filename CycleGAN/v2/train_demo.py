import tensorflow as tf
import matplotlib.pyplot as plt
from models import build_generator, build_discriminator
from dataset import get_dataset, show_batch

# 1️⃣ 构建模型
G = build_generator()   # X → Y
F = build_generator()   # Y → X
D_X = build_discriminator()  # 判别 X 域
D_Y = build_discriminator()  # 判别 Y 域

# 2️⃣ 定义优化器
G_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
D_X_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
D_Y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
L1 = lambda a, b: tf.reduce_mean(tf.abs(a - b))
lambda_cyc = 10.0
lambda_id = 0.5 * lambda_cyc

# 3️⃣ 训练步
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        # 生成假样本
        fake_y = G(real_x, training=True)
        fake_x = F(real_y, training=True)

        # 重建循环
        cyc_x = F(fake_y, training=True)
        cyc_y = G(fake_x, training=True)

        # 身份映射
        same_x = F(real_x, training=True)
        same_y = G(real_y, training=True)

        # 判别器输出
        D_X_real = D_X(real_x, training=True)
        D_X_fake = D_X(fake_x, training=True)
        D_Y_real = D_Y(real_y, training=True)
        D_Y_fake = D_Y(fake_y, training=True)

        # GAN loss
        G_GAN_loss = bce(tf.ones_like(D_Y_fake), D_Y_fake)
        F_GAN_loss = bce(tf.ones_like(D_X_fake), D_X_fake)

        # Cycle & Identity losses
        cycle_loss = L1(cyc_x, real_x) + L1(cyc_y, real_y)
        id_loss = L1(same_x, real_x) + L1(same_y, real_y)

        # 总生成器 & 判别器损失
        G_total_loss = G_GAN_loss + F_GAN_loss + lambda_cyc * cycle_loss + lambda_id * id_loss
        D_X_loss = bce(tf.ones_like(D_X_real), D_X_real) + bce(tf.zeros_like(D_X_fake), D_X_fake)
        D_Y_loss = bce(tf.ones_like(D_Y_real), D_Y_real) + bce(tf.zeros_like(D_Y_fake), D_Y_fake)

    # 计算梯度
    G_grads = tape.gradient(G_total_loss, G.trainable_variables + F.trainable_variables)
    D_X_grads = tape.gradient(D_X_loss, D_X.trainable_variables)
    D_Y_grads = tape.gradient(D_Y_loss, D_Y.trainable_variables)

    # 应用梯度
    G_optimizer.apply_gradients(zip(G_grads, G.trainable_variables + F.trainable_variables))
    D_X_optimizer.apply_gradients(zip(D_X_grads, D_X.trainable_variables))
    D_Y_optimizer.apply_gradients(zip(D_Y_grads, D_Y.trainable_variables))

    del tape
    return G_total_loss, D_X_loss, D_Y_loss

# 4️⃣ 训练循环
trainX = get_dataset('data/butterfly_real/*.jpeg', batch_size=2)
trainY = get_dataset('data/butterfly_origami/*.jpg', batch_size=2)

for epoch in range(3):  # 试跑3个epoch
    for real_x, real_y in zip(trainX, trainY):
        G_loss, DX_loss, DY_loss = train_step(real_x, real_y)
    print(f"Epoch {epoch+1}: G={G_loss:.3f} DX={DX_loss:.3f} DY={DY_loss:.3f}")

    # 展示生成结果
    sample = next(iter(trainX))
    fake_y = G(sample, training=False)
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.imshow((sample[0]+1)/2); plt.title("Real X"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow((fake_y[0]+1)/2); plt.title("Fake Y"); plt.axis("off")
    plt.show()
