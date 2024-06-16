import jax.numpy as jnp
from jax import random
from NeuralNet import NeuralNet
from jax import grad
import pickle
import numpy as np
import NN
from functions import MSE

class VariationalAutoencoder:
    def _init_locals(self):
        encoder_params = self.encoder.params
        decoder_params = self.decoder.params
        mean_encoder_params = self.mean_encoder.params
        var_encoder_params = self.var_encoder.params
        self.encoder.params = encoder_params
        self.decoder.params = decoder_params
        self.mean_encoder.params = mean_encoder_params
        self.var_encoder.params = var_encoder_params
        self.encoder._init_locals()
        self.decoder._init_locals()
        self.mean_encoder._init_locals()
        self.var_encoder._init_locals()

        def forward(params, x):
            encoder_params, decoder_params, mean_encoder_params, var_encoder_params = params
            encoded = self.encoder.forward(encoder_params, x)
            mu = self.mean_encoder.forward(mean_encoder_params, encoded)
            log_var = self.var_encoder.forward(var_encoder_params, encoded)
            std = jnp.exp(0.5 * log_var)
            eps = random.normal(self.rng, mu.shape)
            z = mu + eps * std
            decoded = self.decoder.forward(decoder_params, z)
            return decoded, mu, log_var

        def loss(params, x, y):
            y_hat, mu, log_var = forward(params, x)
            kl_div = -0.5 * jnp.sum(1 + log_var - jnp.square(mu) - jnp.exp(log_var), axis=1)
            total_loss = self.loss_f(y_hat, y) + jnp.mean(kl_div) * 0.02
            return total_loss

        self.forward = forward
        self.loss = loss

    def __init__(self, encoder_layers, encoder_layer_shapes, decoder_layers, decoder_layer_shapes, loss_f):
        self.encoder = NeuralNet(encoder_layers, encoder_layer_shapes, loss_f)
        self.decoder = NeuralNet(decoder_layers, decoder_layer_shapes, loss_f)
        self.mean_encoder = NeuralNet([
            NN.LayerMatMul,
            NN.LReLU,
        ], [
            (256, 128), ()
        ], loss_f)
        self.var_encoder = NeuralNet([
            NN.LayerMatMul,
            NN.LReLU,
        ], [
            (256, 128), ()
        ], loss_f)
        self.loss_f = loss_f
        self.params = (self.encoder.params, self.decoder.params, self.mean_encoder.params, self.var_encoder.params)
        self.rng = random.PRNGKey(0)
        self._init_locals()
        print(f"VariationalAutoencoder initialized with {len(self.params[0])} encoder layers and {len(self.params[1])} decoder layers")

    def __call__(self, x):
        return self.forward(self.params, x)

    def update(self, x, y, learning_rate):
        grads = grad(self.loss)(self.params, x, y)
        enc_grads, dec_grads, mean_grads, var_grads = grads

        self.encoder.updateWithGrad(enc_grads, learning_rate)
        self.decoder.updateWithGrad(dec_grads, learning_rate)
        self.mean_encoder.updateWithGrad(mean_grads, learning_rate)
        self.var_encoder.updateWithGrad(var_grads, learning_rate)
        self.params = (self.encoder.params, self.decoder.params, self.mean_encoder.params, self.var_encoder.params)

    def train(self, X, y, val_x, val_y, epochs=100, learning_rate=0.01):
        # implement mini-batch training
        batch_size = 512
        for epoch in range(epochs):

            for i in range(0, len(X), batch_size):
                x_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.update(x_batch, y_batch, learning_rate)
            loss = self.loss(self.params, val_x, val_y)
            print(f'Epoch: {epoch}, Loss: {loss}')

    def save(self, path):
        # save model into the file
        with open(path, 'wb') as f:
            pickle.dump((self.params, self.encoder.layers, self.decoder.layers, self.mean_encoder.layers, self.var_encoder.layers, self.loss_f), f)
            

    def load(self, path):
        with open(path, 'rb') as f:
            self.params, self.encoder.layers, self.decoder.layers, self.mean_encoder.layers, self.var_encoder.layers, self.loss_f = pickle.load(f)
            self._init_locals()