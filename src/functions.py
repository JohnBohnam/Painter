from jax import numpy as jnp

def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def MSE(y_hat, y):
    return jnp.mean((y_hat - y) ** 2)

def log_entropy(y_hat, y):
    return -jnp.mean(y_hat * y)

def cross_entropy(y_hat, y):
    eps = 1e-10
    return -jnp.mean(y * jnp.log(y_hat+eps))

