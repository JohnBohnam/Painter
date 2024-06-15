from jax import random, grad
import pickle
import jax.numpy as jnp

class NeuralNet:
    def _init_locals(self):
        def forward(params, x):
            for i in range(len(self.layers)):
                x = self.layers[i].forward(params[i], x)
            return x
    
        def loss(params, x, y):
            y_hat = forward(params, x)
            return self.loss_f(y_hat, y)
        
        self.forward = forward
        self.loss = loss
    
    def __init__(self, layers, layer_shapes, loss_f):
        self.layers = layers
        self.layer_shapes = layer_shapes
        self.loss_f = loss_f
        self.params = []
        self.rng = random.PRNGKey(25)

        for i in range(len(layer_shapes)):
            self.params.append(layers[i].init_params(self.rng, layer_shapes[i]))

        # def forward(params, x):
        #     for i in range(len(self.layers)):
        #         # print(f'Layer {i} - {self.layers[i]}: {params[i].shape}, {x.shape}')
        #         x = self.layers[i].forward(params[i], x)
        #     return x

        # self.forward = forward

        # def loss(params, x, y):
        #     y_hat = forward(params, x)
        #     return self.loss_f(y_hat, y)

        # self.loss = loss
        
        self._init_locals()

        print(f"NN initialized with {len(self.params)} layers")

    def __call__(self, x):
        return self.forward(self.params, x)

    def updateWithGrad(self, grads, learning_rate):
        for i in range(len(self.params)):
            if grads[i].shape == (0,):
                continue

            g_max = jnp.max(jnp.abs(grads[i]))
            self.params[i] = self.params[i] - learning_rate * grads[i]
            # ensure that the weights are not negative

    def update(self, x, y, learning_rate):
        grads = grad(self.loss)(self.params, x, y)
        self.updateWithGrad(grads, learning_rate)

    def train(self, X, y, epochs=100, learning_rate=0.01):
        for _ in range(epochs):
            self.update(X, y, learning_rate)
            loss = self.loss(self.params, X, y)
            print(f'Epoch: {_}, Loss: {loss}')
            
    def save(self, path):
        # save model into the file
        with open(path, 'wb') as f:
            pickle.dump((self.params, self.layers, self.layer_shapes, self.loss_f), f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.params, self.layers, self.layer_shapes, self.loss_f = pickle.load(f)
            self._init_locals()