# The goal is to write a two neural netwroks:
# - One that is taking an image an returning a vector in the latent space.
# - One that is taking a vector in the latent space and returning an image.
# Use tensorflow or pytorch for the encoder and decoder.
# Test the network on the images in data folder.
# use fully connected layer to get the latent space

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from typing import List, Tuple
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random as jrandom
from jax import lax
from jax import nn
import tensorflow as tf
import datetime

def load_images(path: str, limit=None) -> List[np.ndarray]:
    images = []
    image_count = 0
    for file in os.listdir(path):
        if image_count == limit:
            break
        image = Image.open(os.path.join(path, file)).resize((256, 256)).convert('RGB')
        image = np.array(image)
        images.append(image)
        image_count += 1
    return images

def save_images(images: List[np.ndarray], path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    for i, image in enumerate(images):
        image = Image.fromarray(image)
        image.save(os.path.join(path, f'{i+1}.png'))

def preprocess_images(images: List[np.ndarray]) -> np.ndarray:
    images = np.array(images)
    images = images / 255.0
    images = images.reshape(images.shape[0], -1)
    return images

def postprocess_images(images: np.ndarray) -> List[np.ndarray]:
    images = images.reshape(images.shape[0], 256, 256, 3)
    images = images * 255.0
    images = images.astype(np.uint8)
    return images

def forward(encoder, decoder, x: np.ndarray) -> np.ndarray:
    latent = encoder(x)
    return decoder(latent)

def backward(encoder, decoder, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> None:
    with tf.GradientTape() as tape:
        latent = encoder(x)
        reconstructed = decoder(latent)
        loss = tf.reduce_mean(tf.square(y - reconstructed))
        print(f'Loss: {loss}')
    gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    decoder.optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    encoder.optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))

def train(encoder, decoder, x: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01, batch_size: int = 32) -> None:
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        idx = np.random.choice(x.shape[0], size=batch_size, replace=False)
        x_batch = x[idx]
        y_batch = y[idx]
        backward(encoder, decoder, x_batch, y_batch, lr)

def train2(encoder, decoder, dataset, epochs: int = 100, lr: float = 0.01) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                encoded = encoder(x_batch, training=True)
                decoded = decoder(encoded, training=True)
                loss = mse_loss_fn(y_batch, decoded)
            
            grads = tape.gradient(loss, encoder.trainable_weights + decoder.trainable_weights)
            optimizer.apply_gradients(zip(grads, encoder.trainable_weights + decoder.trainable_weights))
            
            if step % 100 == 0:
                print(f'Step {step}: loss = {loss.numpy()}')


def save(encoder, decoder, path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    encoder.save(os.path.join(path, 'encoder.keras'))
    decoder.save(os.path.join(path, 'decoder.keras'))

def load(encoder, decoder, path: str) -> None:
    encoder.load(os.path.join(path, 'encoder.keras'))
    decoder.load(os.path.join(path, 'decoder.keras'))

def main():
    images = load_images('data', 100)
    images = preprocess_images(images)
    images = images.reshape(images.shape[0], 256, 256, 3)
    
    dataset = tf.data.Dataset.from_tensor_slices((images, images))
    dataset = dataset.shuffle(buffer_size=1024).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # define encoder and decoder using convolutional neural networks
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(256, 256, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu')
    ])

    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32*32*3, activation='relu'),
        tf.keras.layers.Reshape((32, 32, 3)),
        tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Reshape((256, 256, 3))
    ])
    optimizer = tf.keras.optimizers.Adam()  

    # use optimizer to compile the model
    encoder.compile(optimizer=optimizer)
    decoder.compile(optimizer=optimizer)

    # train the model
    train2(encoder, decoder, dataset, epochs=100, lr=0.01)

    # test the model
    latent = encoder(images)
    reconstructed = decoder(latent)

    reconstructed = postprocess_images(reconstructed.numpy())

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    save_images(reconstructed, f'reconstructed_{current_datetime}')

    save(encoder, decoder, f'models_{current_datetime}')

if __name__ == '__main__':
    main()