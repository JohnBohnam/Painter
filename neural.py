import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from typing import List
import jax
import jax.numpy as jnp
from jax import grad, jit, random

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

def init_params(rng, input_dim, layer_sizes):
    params = []
    for i, size in enumerate(layer_sizes):
        rng, layer_rng = random.split(rng)
        input_size = input_dim if i == 0 else layer_sizes[i-1]
        params.append([
            random.normal(layer_rng, (input_size, size)) * jnp.sqrt(2.0 / input_size),
            jnp.zeros(size)
        ])
    return params

def forward(params, x):
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = jax.nn.relu(x)
    w, b = params[-1]
    x = jnp.dot(x, w) + b
    return x

def compute_loss(encoder_params, decoder_params, x):
    latent = forward(encoder_params, x)
    reconstructed = forward(decoder_params, latent)
    loss = jnp.mean((x - reconstructed) ** 2)
    return loss

@jit
def update_params(params, grads, learning_rate):
    return [(w - learning_rate * dw, b - learning_rate * db) for (w, b), (dw, db) in zip(params, grads)]

@jit
def train_step(encoder_params, decoder_params, batch, learning_rate):
    loss, grads = jax.value_and_grad(compute_loss, argnums=(0, 1))(encoder_params, decoder_params, batch)
    encoder_grads, decoder_grads = grads
    encoder_params = update_params(encoder_params, encoder_grads, learning_rate)
    decoder_params = update_params(decoder_params, decoder_grads, learning_rate)
    return encoder_params, decoder_params, loss

def train(encoder_params, decoder_params, data, epochs=100, batch_size=32, learning_rate=0.001):
    num_batches = data.shape[0] // batch_size
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(num_batches):
            batch = data[i*batch_size:(i+1)*batch_size]
            encoder_params, decoder_params, loss = train_step(encoder_params, decoder_params, batch, learning_rate)
            epoch_loss += loss
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / num_batches}')
    return encoder_params, decoder_params

def main():
    images = load_images('data', 300)
    images = preprocess_images(images)

    rng = random.PRNGKey(0)

    encoder_layer_sizes = [128, 64, 32]
    decoder_layer_sizes = [64, 128, images.shape[1]]

    encoder_params = init_params(rng, images.shape[1], encoder_layer_sizes)
    decoder_params = init_params(rng, encoder_layer_sizes[-1], decoder_layer_sizes)

    encoder_params, decoder_params = train(encoder_params, decoder_params, images, epochs=100, batch_size=32, learning_rate=0.001)

    latent = jax.vmap(lambda x: forward(encoder_params, x))(images)
    reconstructed = jax.vmap(lambda z: forward(decoder_params, z))(latent)

    reconstructed = postprocess_images(np.array(reconstructed))

    save_images(reconstructed, 'reconstructed')

if __name__ == '__main__':
    main()
