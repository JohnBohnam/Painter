from jax import numpy as jnp
from jax import random
import datatransform
import NN
from NeuralNet import NeuralNet
from functions import one_hot, MSE


def test1():
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

def test2():
    def fun(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2
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

def conv_test():
    size = 28

    rng = random.PRNGKey(0)
    X = random.normal(rng, (1000, size, size))
    X = X.reshape((1000, size, size, 1))
    X = X.transpose((0, 3, 1, 2))

    def fun(x):
        return jnp.sum(jnp.abs(x))

    y = jnp.array([[fun(x)] for x in X])

    layers = [
        NN.LayerConv2D,
        NN.LReLU,
        NN.LayerConv2D,
        NN.LayerFlatten,
        NN.LReLU,
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
    ]

    n_channels = 3
    layer_shapes = [
        (3, 3, 1, 1),
        (),
        (3, 3, 1, 1),
        (),
        (),
        (size * size, 1),
        (1,),
        (),
    ]

    model = NN.NeuralNet(layers, layer_shapes, MSE)

    model.train(X, y, epochs=1000, learning_rate=0.01)
    # model.forward(model.params, X)

if __name__ == '__main__':
    test1()
    test2()
    conv_test()