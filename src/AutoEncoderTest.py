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

def autoencoder_test():
    # print current working directory
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
    n_train = 60000

    val_images = train_images[int(n_train * 0.6):int(n_train * 0.8)]
    val_labels = train_labels[int(n_train * 0.6):int(n_train * 0.8)]

    test_images = train_images[int(n_train * 0.8):]
    test_labels = train_labels[int(n_train * 0.8):]

    train_images = train_images[:int(n_train * 0.6)]
    train_labels = train_labels[:int(n_train * 0.6)]

    print(train_images.shape)

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
        (28 * 28 * 16, 32),
        (32,),
        (),
        (32, 16),
        (16,),
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
        (16, 16 * 28 * 28),
        (16 * 28 * 28,),
        (),
        (),
        (3, 3, 16, 8),
        (),
        (3, 3, 8, 4),
        (),
        (3, 3, 4, 1),
        (),

    ]

    model = AutoEncoder(encoder_layers, encoder_layer_shapes, decoder_layers, decoder_layer_shapes, MSE)

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

    print('Shape: ', train_images.shape)
    
    # out_before = model(train_images)
    # model.save('../models/autoencoder.model')
    # model2 = AutoEncoder(encoder_layers, encoder_layer_shapes, decoder_layers, decoder_layer_shapes, MSE)
    # model2.load('../models/autoencoder.model')
    # out_after = model2(train_images)
    
    # assert jnp.allclose(out_before, out_after)
    # print('Model loaded successfully')

    model.train(train_images, train_images, val_images, val_images, epochs=20, learning_rate=0.1)

    model.save('../models/autoencoder.model')


    test_images = model(test_images)

    test_images = test_images * 255
    test_images = jnp.reshape(test_images, (len(test_images), 28, 28))

    datatransform.save_images(test_images, '../output')

   
if __name__ == '__main__':
    autoencoder_test()





