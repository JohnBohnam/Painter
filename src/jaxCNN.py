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
        self.rng = random.PRNGKey(25)
        
        for i in range(len(layer_shapes)):
            self.params.append(layers[i].init_params(self.rng, layer_shapes[i]))

        def forward(params, x):
            for i in range(len(self.layers)):
                # print(f'Layer {i} - {self.layers[i]}: {params[i].shape}, {x.shape}')
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------

class AutoEncoder:
    def __init__(self, encoder_layers, encoder_layer_shapes, decoder_layers, decoder_layer_shapes, loss_f):
        self.encoder = NeuralNet(encoder_layers, encoder_layer_shapes, loss_f)
        self.decoder = NeuralNet(decoder_layers, decoder_layer_shapes, loss_f)
        self.loss_f = loss_f
        self.params = (self.encoder.params, self.decoder.params)

        def forward(params, x):
            # print('Forward')
            encoder_params, decoder_params = params
            encoded = self.encoder.forward(encoder_params, x)
            decoded = self.decoder.forward(decoder_params, encoded)
            return decoded
        
        self.forward = forward

        def loss(params, x, y):
            # enc_params_len = len(self.encoder.params)
            # self.encoder.params = params[:enc_params_len]
            # self.decoder.params = params[enc_params_len:]

            y_hat = self.forward(params, x)
            return self.loss_f(y_hat, y)
        
        self.loss = loss

    def __call__(self, x):
        return self.forward(self.params, x)

    def update(self, x, y, learning_rate):
        # combined_params = (self.encoder.params + self.decoder.params)
        grads = grad(self.loss)(self.params, x, y)

        # enc_params_len = len(self.encoder.params)
        # enc_grads = grads[:enc_params_len]
        # dec_grads = grads[enc_params_len:]
        enc_grads, dec_grads = grads

        self.encoder.updateWithGrad(enc_grads, learning_rate)
        self.decoder.updateWithGrad(dec_grads, learning_rate)
        self.params = (self.encoder.params, self.decoder.params)
        #print(self.params)

    def train(self, X, y, epochs=100, learning_rate=0.01):
        # implement mini-batch training
        batch_size = 512
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                x_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                self.update(x_batch, y_batch, learning_rate)
            loss = self.loss(self.params, X, y)
            print(f'Epoch: {epoch}, Loss: {loss}')

    def save(self, path):
        # save model into the file
        import pickle
        with open(path, 'wb') as f:
            pickle.dump((self.params,   self.encoder.layers, self.encoder.forward, self.encoder.loss, \
                                        self.decoder.layers, self.decoder.forward, self.decoder.loss), f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.params, self.encoder.layers, self.encoder.forward, self.encoder.loss, \
             self.decoder.layers, self.decoder.forward, self.decoder.loss = pickle.load(f)




# not done yet
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
        (size*size, 1),
        (1,),
        (),
    ]
    
    model = NN.NeuralNet(layers, layer_shapes, MSE)
    
    model.train(X, y, epochs=1000, learning_rate=0.01)
    # model.forward(model.params, X)

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
    n_train = 10000
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
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
        (3,3,1,8),
        (),
        (3,3,8,16),
        (),
        (),
        (28*28*16, 32),
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
        (16, 16*28*28),
        (16*28*28,),
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
    train_images = train_images.transpose((0, 3, 1, 2))
    test_images = test_images.transpose((0, 3, 1, 2))

    # convert train_image elements to float32
    train_images = train_images.astype(jnp.float32)
    test_images = test_images.astype(jnp.float32)
    train_images = train_images / 255
    test_images = test_images / 255
    
    print('Shape: ', train_images.shape)

    model.train(train_images, train_images, epochs=5, learning_rate=0.1)

    # TODO
    # Save is not working properly. If you could fix it I would be grateful.
    # model.save('../models/autoencoder.model')

    test_images = model(test_images)

    test_images = test_images * 255
    test_images = jnp.reshape(test_images, (len(test_images), 28, 28))

    datatransform.save_images(test_images, '../output')



if __name__ == "__main__":
    # jax.config.update("jax_traceback_filtering", "off")
    # main()
    # simple()
    # conv_test()
    autoencoder_test()
    
    