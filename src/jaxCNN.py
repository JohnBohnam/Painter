import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, random
import datatransform
from jax.scipy.special import logsumexp

import NN

def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)



    
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
    size = 10
    
    rng = random.PRNGKey(0)
    X = random.normal(rng, (1000, size, size))
    X = X.reshape((1000, size, size, 1))
    X = X.transpose((0, 3, 1, 2))
    
    def fun(x):
        return jnp.sum(jnp.abs(x))
    
    y = jnp.array([[fun(x)] for x in X])
    
    layers = [
        NN.LayerConv2D,
        NN.LayerBias,
        NN.LayerConv2D,
        NN.LayerBias,
        NN.LayerFlatten,
        NN.LReLU,
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
    ]
    
    n_channels = 1
    layer_shapes = [
        # usage: (kernel_size, kernel_size, input_channels, output_channels)
        (2, 2, 1, n_channels),
        (n_channels, size, size),
        (2, 2, n_channels, 1),
        (1, size, size),
        (),
        (),
        # the output size is just input_h * input_w * input_channels
        (size*size*1, 1), 
        (1,),
        (),
    ]
    
    model = NN.NeuralNet(layers, layer_shapes, MSE)
    
    model.train(X, y, epochs=1000, learning_rate=0.01)
    # model.forward(model.params, X)
    
    n_show = 2
    idx = random.choice(rng, len(X), (n_show,), replace=False)
    
    print(f"input: \n{X[idx]}, \noutput: \n{model(X[idx])}, \nexpected: \n{y[idx]}")
        

if __name__ == "__main__":
    # main()
    # simple()
    conv_test()
    
    
    