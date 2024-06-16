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
from functions import binary_cross_entropy
from functions import one_hot

def vae_test():
    training_images_filepath = '../data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte'
    training_labels_filepath = '../data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_filepath = '../data/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = '../data/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    mnist_dataloader = datatransform.MnistDataloader(training_images_filepath, training_labels_filepath,
                                                     test_images_filepath, test_labels_filepath)
    (train_images, train_labels), (test_images, test_labels) = mnist_dataloader.load_data()

    n_train = 60000
    val_images = train_images[int(n_train *0.6):int(n_train *0.8)]
    val_labels = train_labels[int(n_train *0.6):int(n_train *0.8)]
    test_images = train_images[int(n_train *0.8):]
    test_labels = train_labels[int(n_train *0.8):]
    train_images = train_images[:int(n_train *0.6)]
    train_labels = train_labels[:int(n_train *0.6)]

    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    num_pixels = 28 * 28

    train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    val_images = jnp.reshape(val_images, (len(val_images), num_pixels))

    num_labels = 10

    train_labels = one_hot(train_labels, num_labels)

    encoder_layers = [
        NN.LayerFlatten,
        NN.LayerMatMulDropout,
        NN.LReLU,
        NN.LayerBias,
        NN.LayerMatMulDropout,
        NN.LReLU,
    ]

    encoder_layer_shapes = [
        (),
        (28 * 28, 512),
        (),
        (512,),
        (512, 256),
        (),
    ]

    decoder_layers = [
        NN.LayerMatMulDropout,
        NN.LReLU,
        NN.LayerBias,
        NN.LayerMatMulDropout,
        NN.LReLU,
        NN.LayerMatMulDropout,
        NN.LReLU,
        NN.Layer2DReshape1,
        NN.LayerConv2D,
        NN.LayerSigmod

    ]

    decoder_layer_shapes = [
        (128, 256),
        (),
        (256,),
        (256, 512),
        (),
        (512, 28*28),
        (),
        (),
        (2, 2, 1, 1),
        (),

    ]

    model = VariationalAutoencoder(encoder_layers, encoder_layer_shapes, decoder_layers, decoder_layer_shapes, MSE)
    #model.load('../models/vae.model')

    train_images = train_images.reshape((len(train_images), 28, 28, 1))
    test_images = test_images.reshape((len(test_images), 28, 28, 1))
    val_images = val_images.reshape((len(val_images), 28, 28, 1))
    train_images = train_images.transpose((0, 3, 1, 2))
    test_images = test_images.transpose((0, 3, 1, 2))
    val_images = val_images.transpose((0, 3, 1, 2))

    # convert train_image elements to float32
    train_images = train_images.astype(jnp.float32)
    test_images = test_images.astype(jnp.float32)
    val_images = val_images.astype(jnp.float32)
    train_images = train_images / 255
    test_images = test_images / 255
    val_images = val_images / 255

    # print('Shape: ', train_images.shape)

    model.train(train_images, train_images, val_images, val_images, epochs=15, learning_rate=0.1)

    model.save('../models/vae.model')

    test_images, _, _ = model(train_images)
    print(MSE(test_images, train_images))
    print(MSE(test_images[0], train_images[0]))
    test_images = test_images * 255


    test_images = jnp.reshape(test_images, (len(test_images), 28, 28))

    datatransform.save_images(test_images, '../output/vae')

if __name__ == '__main__':
    vae_test()
