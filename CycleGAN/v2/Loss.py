import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False) #交叉熵函数
L1 = lambda a, b: tf.reduce_mean(tf.abs(a - b)) #L1范数
lambda_cyc = 10.0
lambda_id = 0.5 * lambda_cyc

def train_step(real_x, real_y):
    with tf.GradientTape(persistent = True) as tape:
        #forward prop
        fake_y = G(real_x, training = True)
        fake_x = F(real_y, training = True)
        cyc_x = F(fake_y, training = True)
        cyc_y = G(fake_x, training = True)
        same_x = F(real_x, training = True)
        same_y = G(real_y, training = True)

        D_X_real = D_X(real_x, training = True)
        D_X_fake = D_X(fake_x, training = True)
        D_Y_real = D_Y(real_y, training = True)
        D_Y_fake = D_Y(fake_y, training = True)

        G_GAN_loss = bce(tf.ones_like(D_Y_fake), D_Y_fake)
        F_GAN_loss = bce(tf.ones_like(D_X_fake), D_X_fake)

        cycle_loss = L1(cyc_x, real_x) + L1(cyc_y, real_y)

        id_loss = L1(same_x, real_x) + L1(same_y, real_y)

        #生成器总损失
        G_total_loss = G_GAN_loss + F_GAN_loss + lambda_cyc * cycle_loss + lambda_id * id_loss

        #判别器损失
        D_X_loss = bce(tf.ones_like(D_X_real), D_X_real) + bce(tf.zeros_like(D_X_fake), D_X_fake)
        D_Y_loss = bce(tf.ones_like(D_Y_real), D_Y_real) + bce(tf.zeros_like(D_Y_fake), D_Y_fake)

        G_grads = tape.gradient(G_total_loss, G.trainable_variables + F.trainable_variables)
        D_X_grads = tape.gradient(D_X_loss, D_X.trainable_variables)
        D_Y_grads = tape.gradient(D_Y_loss, D_Y.trainable_variables)

        G_optimizer.apply_gradients(zip(G_grads, G.trainable_variables + F.trainable_variables))
        D_X_optimizer.apply_gradients(zip(D_X_grads, D_X.trainable_variables))
        D_Y_optimizer.apply_gradients(zip(D_Y_grads, D_Y.trainable_variables))

        del tape

        return {
            "G_total": G_total_loss,
            "D_X": D_X_loss,
            "D_Y": D_Y_loss
        }