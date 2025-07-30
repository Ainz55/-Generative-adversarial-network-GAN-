import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU доступен")
else:
    print("GPU не найден")


output_dir = "generated_images_tf"
os.makedirs(output_dir, exist_ok=True)


# Генератор
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(28 * 28, activation='tanh'),
        layers.Reshape((28, 28, 1))
    ])
    return model


def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)), # а 2д 1д ба алиш мукунат
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


class GAN:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                                   metrics=['accuracy'])
        self.discriminator.trainable = False
        noise = layers.Input(shape=(latent_dim,))
        fake_image = self.generator(noise)
        validity = self.discriminator(fake_image)
        self.combined = tf.keras.Model(noise, validity)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    def train(self, epochs, batch_size=128):
        (X_train, _), _ = tf.keras.datasets.mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=-1)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_images = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_images = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(real_images, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            if (epoch + 1) % 1000 == 0:
                print(
                    f"Epoch {epoch + 1} D loss: {d_loss[0]:.4f}, G loss: {g_loss:.4f}")
                self.generate_and_save_images(epoch + 1)

    def generate_and_save_images(self, epoch, num=10):
        noise = np.random.normal(0, 1, (num, self.latent_dim))
        generated_images = self.generator.predict(noise)
        generated_images = (generated_images + 1) / 2.0
        fig, axes = plt.subplots(1, num, figsize=(10, 2))
        for i in range(num):
            axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
            axes[i].axis('off')
        plt.savefig(f"{output_dir}/epoch_{epoch}.png")
        plt.close()

# Запуск обучения
latent_dim = 100
gan = GAN(latent_dim)
gan.train(epochs=10000, batch_size=128)
