import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, random
import datatransform
from jax.scipy.special import logsumexp

import NN

def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


class NeuralNet:
    def __init__(self, layers, layer_shapes, loss_f): 
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
    

    
def MSE(y_hat, y):
    return jnp.mean((y_hat - y)**2)
        
def log_entropy(y_hat, y):
    return -jnp.mean(y_hat * y)

def main():
    train_images, train_labels, test_images, test_labels = datatransform.readMNIST('./../data/MNIST/')
    num_pixels = 28 * 28

    train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    
    num_labels = 10

    train_labels = one_hot(train_labels, num_labels)
    
    layers = [
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
    ]
    
    layer_shapes = [
        (784, 200),
        (200,),
        (),
        (200, 10),
        (10,),
        (),
    ]
    

    
    model = NeuralNet(layers, layer_shapes, MSE)
    
    model.train(train_images, train_labels, epochs=100, learning_rate=0.01)
    
    test_images = model(test_images)
    test_labels_hat = jnp.argmax(test_images, axis=1)
    
    accuracy = jnp.mean(test_labels == test_labels_hat)
    print(f'Accuracy: {accuracy}')
    

def simple():
    def fun(x):
        return x[0]**2 + x[1]**2 + x[2]**2
        # return x[0] + 2

    layers = [
        # NN.LayerConv2D,
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
    ]
    
    layes_shapes = [
        (3, 10),
        (10,),
        (),
        (10, 1),
        (1,),
        (),
    ]
    
    model = NeuralNet(layers, layes_shapes, MSE)
    
    rng = random.PRNGKey(0)
    X = random.normal(rng, (1000, 3))
    y = jnp.array([[fun(x)] for x in X])
    
    # X = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # y = jnp.array([[fun(x)] for x in X])
    
    model.train(X, y, epochs=1000, learning_rate=0.05)
    
    n_show = 10
    idx = random.choice(rng, len(X), (n_show,), replace=False)
    print(f"input: \n{X[idx]}, \noutput: \n{model(X[idx])}, \nexpected: \n{y[idx]}")
    
    
# not done yet
def conv_test():
    size = 5
    
    rng = random.PRNGKey(0)
    X = random.normal(rng, (1000, size, size))
    X = X.reshape((1000, size, size, 1))
    
    def fun(x):
        return jnp.sum(x)
    
    y = jnp.array([[fun(x)] for x in X])
    
    layers = [
        NN.LayerConv2D,
        NN.LayerFlatten,
        NN.LReLU,
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
    ]
    
    layer_shapes = [
        (5, 5, 1, 1),
        (),
        (),
        (25, 1),
        (1,),
        (),
    ]
    
    model = NeuralNet(layers, layer_shapes, MSE)
    
    model.train(X, y, epochs=1000, learning_rate=0.05)
    # model.forward(model.params, X)
        

if __name__ == "__main__":
    # main()
    # simple()
    conv_test()
    
    
    