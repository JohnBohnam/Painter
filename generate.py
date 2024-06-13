# Path: Painter/generate.py

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

def main():
    decoder = tf.keras.models.load_model('models/decoder.keras')

    latent_size = 32

    for i in range(10):
        latent = np.random.randn(1, latent_size)
        image = decoder.predict(latent)
        image = image.reshape(256, 256, 3)
        image = image * 255.0
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        if not os.path.exists('generated_images'):
            os.makedirs('generated_images')
        image.save(f'generated_images/{i+1}.png')

if __name__ == '__main__':
    main()