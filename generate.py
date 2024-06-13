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
    decoder = tf.keras.models.load_model('models/mnist/decoder.keras')

    latent_size = 32

    for i in range(10):
        latent = np.random.randn(1, latent_size)
        image = decoder.predict(latent)
        image = image.reshape(28, 28)
        image = image * 255.0
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        if not os.path.exists('generated_images/mnist'):
            os.makedirs('generated_images/mnist')
        image.save(f'generated_images/mnist/{i+1}.png')

if __name__ == '__main__':
    main()