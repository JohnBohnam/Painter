import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, random

import datatransform
from jax.scipy.special import logsumexp
from typing import List, Tuple

class Layer:
    def forward(params, x):
        raise NotImplementedError()
    
    def init_params(rng, shape):
        return jnp.array([])

class LayerMatMul(Layer):
    def forward(params, x):
        return jnp.dot(x, params)
    
    def init_params(rng, shape):
        input_shape, output_shape = shape
        W = random.normal(rng, (input_shape, output_shape))*jnp.sqrt(2.0/input_shape)
        return W
    
class LayerBias(Layer):
    def forward(params, x):
        return x + params
    
    def init_params(rng, shape):
        return jnp.zeros(shape)

class LReLU(Layer):
    def forward(params, x):
        return jax.nn.leaky_relu(x)
    
    
class LayerConv2D(Layer):
    def forward(params, x):
        out = jax.lax.conv(x, 
                           jnp.transpose(params, [3, 2, 0, 1]),
                           (1, 1),
                           'SAME')
        return out
    
    def init_params(rng, shape):
        # return random.normal(rng, shape)
        return jnp.zeros(shape)
    
class LayerFlatten(Layer):
    def forward(params, x):
        return jnp.reshape(x, (x.shape[0], -1))
    
    


class NeuralNet:
    def __init__(self, layers: List[Layer], layer_shapes: List[Tuple[int]], loss_f): 
        self.layers = layers
        self.layer_shapes = layer_shapes
        self.loss_f = loss_f
        self.params = []
        self.rng = random.PRNGKey(100)
        
        for i in range(len(layer_shapes)):
            self.params.append(layers[i].init_params(self.rng, layer_shapes[i]))
            
        def forward(params, x):
            for i in range(len(self.layers)):
                x = self.layers[i].forward(params[i], x)
            return x
        
        self.forward = forward
        
        def loss(params, x, y):
            y_hat = forward(params, x)
            return self.loss_f(y_hat, y)
        
        self.loss = loss
        
        print(f"NN initialized with {len(self.params)} layers")
        
    
    def __call__(self, x):
        return self.forward(self.params, x)

    
    def update(self, x, y, learning_rate):
        grads = grad(self.loss)(self.params, x, y)
        for i in range(len(self.params)):
            if grads[i].shape == (0,):
                continue
            
            g_max = jnp.max(jnp.abs(grads[i]))
            if g_max > 1:
                grads[i] = grads[i] / g_max
            # print(f'Layer {i} has shape {self.params[i].shape}')
            # print(f'Layer {i} has grad shape {grads[i].shape}')
            # print(f'grad for layer {i} :\n{grads[i]}')
            self.params[i] = self.params[i] - learning_rate * grads[i]
            
        
    def train(self, X, y, epochs=100, learning_rate=0.01):
        for _ in range(epochs):
            self.update(X, y, learning_rate)
            loss = self.loss(self.params, X, y)
            print(f'Epoch: {_}, Loss: {loss}')
    
