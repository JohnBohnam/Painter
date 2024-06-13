import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, random, lax
import datatransform

class Layer:
    def forward(self, x):
        # Returns the values for the next level of NN
        # x: input to the layer
        raise NotImplementedError()
    
    def backward(self, grad_output, lr):
        # Returns the gradient of the loss with respect to the input
        # Updates the weights and biases
        # grad_output: gradient of the loss with respect to the output of the layer
        # lr: learning rate
        raise NotImplementedError()

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = jax.random.normal(random.PRNGKey(0), (input_size, output_size))
        self.bias = jax.random.normal(random.PRNGKey(0), (output_size,))
        
    def forward(self, x):
        return jnp.dot(x, self.weights) + self.bias # Dimension mismatch
    
    def backward(self, grad_output, lr):
        grad_input = jnp.dot(grad_output, self.weights.T)
        grad_weights = jnp.dot(self.input.T, grad_output)
        grad_bias = jnp.sum(grad_output, axis=0)

        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias
        
        return grad_input

class InputLayer(Layer):
    def forward(self, x):
        return x
    
    def backward(self, grad_output, lr):
        return grad_output

class OutputLayer(Layer):
    def forward(self, x):
        return x
    
    def backward(self, y_pred, y_true, lr):
        return y_pred - y_true # Dimension mismatch

class MaxPoolingLayer(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, x):
        return x

    def backward(self, grad_output, lr):
        return grad_output

class Conv2D(Layer):
    def __init__(self, input_shape, kernel_shape):
        self.weights = jax.random.normal(random.PRNGKey(0), kernel_shape)
        self.bias = jax.random.normal(random.PRNGKey(0), (kernel_shape[-1],))
    
        self.input_shape = input_shape
        self.stride = 1
    
    def forward(self, x):
        return jax.lax.conv(x, self.weights, (self.stride, self.stride), 'SAME') + self.bias # Dimension mismatch
    
    def backward(self, grad_output, lr):
        grad_weights = grad(lambda w: jnp.mean((grad_output - jax.lax.conv(self.input, w, (self.stride, self.stride), self.padding) - self.bias)**2))(self.weights)
        grad_bias = grad(lambda b: jnp.mean((grad_output - jax.lax.conv(self.input, self.weights, (self.stride, self.stride), self.padding) - b)**2))(self.bias)
        
        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias
        
        return grad_output

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x, y, lr):
        y_pred = self.forward(x)

        grad_output = self.layers[-1].backward(y_pred, y, lr) # 
        
        for layer in reversed(self.layers[:-1]):
            grad_output = layer.backward(grad_output, lr)
        
        loss = jnp.mean((y - y_pred) ** 2)
        return loss
    
    def fit(self, x, y, lr, epochs):
        for epoch in range(epochs):
            loss = self.backward(x, y, lr)
            print(f'Epoch {epoch + 1}, Loss: {loss}')

def main():
    print('Reading MNIST data')
    train_images, train_labels, test_images, test_labels = datatransform.readMNIST('./../data/MNIST/')

    print('Preprocessing images')
    
    train_images = train_images / 255.0

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)

    train_labels = datatransform.one_hot_encode(train_labels, 10)

    layers = [
        InputLayer(),
        Conv2D((28, 28, 11), (3, 3, 16)),
        MaxPoolingLayer(2),
        Conv2D((14, 14, 16), (3, 3, 32)),
        MaxPoolingLayer(2),
        DenseLayer(7*7*32, 128),
        DenseLayer(128, 10),
        DenseLayer(10, 28 * 28),
        OutputLayer()
    ]

    model = NeuralNetwork(layers)

    print('Training the model')
    model.fit(train_images, train_labels, 0.01, 10)
    print('Model trained')

    generated_images = model.forward(train_images[:10])

    generated_images = generated_images.reshape(generated_images.shape[0], 28, 28)

    datatransform.postprocess_images(generated_images)

    datatransform.save_images(generated_images, 'generated_images')

if __name__ == '__main__':
    main()
