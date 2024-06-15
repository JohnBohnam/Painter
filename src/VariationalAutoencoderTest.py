from jax import numpy as jnp
from jax import random
from jax import grad
from jax import jit
from jax import vmap
from jax import lax
from jax.scipy.special import logsumexp
import datatransform
import NN
from NeuralNet import NeuralNet
from AutoEncoder import AutoEncoder
from VariationalAutoencoder import VariationalAutoencoder
import numpy as np
from functions import MSE
from functions import one_hot

def vae_test():
    training_images_filepath = '../data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte'
    training_labels_filepath = '../data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_filepath = '../data/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = '../data/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    mnist_dataloader = datatransform.MnistDataloader(training_images_filepath, training_labels_filepath,
                                                     test_images_filepath, test_labels_filepath)
    (train_images, train_labels), (test_images, test_labels) = mnist_dataloader.load_data()
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    num_pixels = 28 * 28

    train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    n_train = 10000
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]

    num_labels = 10

    train_labels = one_hot(train_labels, num_labels)

    encoder_layers = [
        NN.LayerConv2D,
        NN.LReLU,
        NN.LayerConv2D,
        NN.LReLU,
        NN.LayerFlatten,
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
    ]

    encoder_layer_shapes = [
        (3, 3, 1, 8),
        (),
        (3, 3, 8, 16),
        (),
        (),
        (28 * 28 * 16, 256),
        (256,),
        (),
        (256, 128),
        (128,),
        (),
    ]

    decoder_layers = [
        NN.LayerMatMul,
        NN.LayerBias,
        NN.LReLU,
        NN.Layer2DReshape,
        NN.LayerConv2DTranspose,
        NN.LReLU,
        NN.LayerConv2DTranspose,
        NN.LReLU,
        NN.LayerConv2DTranspose,
        NN.LayerSigmod,
    ]

    decoder_layer_shapes = [
        (64, 64 * 28 * 28),
        (64 * 28 * 28,),
        (),
        (),
        (3, 3, 64, 8),
        (),
        (3, 3, 8, 4),
        (),
        (3, 3, 4, 1),
        (),
    ]

    model = VariationalAutoencoder(encoder_layers, encoder_layer_shapes, decoder_layers, decoder_layer_shapes, MSE)
    model.load('../models/vae.model')

    train_images = train_images.reshape((len(train_images), 28, 28, 1))
    test_images = test_images.reshape((len(test_images), 28, 28, 1))
    train_images = train_images.transpose((0, 3, 1, 2))
    test_images = test_images.transpose((0, 3, 1, 2))

    # convert train_image elements to float32
    train_images = train_images.astype(jnp.float32)
    test_images = test_images.astype(jnp.float32)
    train_images = train_images / 255
    test_images = test_images / 255

    # print('Shape: ', train_images.shape)

    model.train(train_images, train_images, epochs=20, learning_rate=0.01)

    model.save('../models/vae.model')

    test_images, _, _ = model(test_images)

    test_images = test_images * 255

    test_images = jnp.reshape(test_images, (len(test_images), 28, 28))

    datatransform.save_images(test_images, '../output/vae')

if __name__ == '__main__':
    vae_test()