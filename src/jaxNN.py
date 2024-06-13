import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, random

import datatransform
from jax.scipy.special import logsumexp

def _predict(params, x):
    for w, b in params[:-1]:
        x = jnp.dot(x, w) + b
        x = jax.nn.relu(x)
    w, b = params[-1]
    x = jnp.dot(x, w) + b
    return x - logsumexp(x)

def _batched_predict(params, X):
    return jax.vmap(lambda x: _predict(params, x))(X)

def accuracy(self, X, y):
    params = list(zip(self.Ws, self.bs))
    y_hat = jnp.argmax(_batched_predict(params, X), axis=1)
    return jnp.mean(y == y_hat)

def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


class NNDense:
    def __init__(self, layer_sizes, loss_f):
        self.layer_sizes = layer_sizes
        self.Ws = []
        self.bs = []
        self.rng = random.PRNGKey(100)
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]
        self.loss_f = loss_f
        
        for i in range(1, len(layer_sizes)):
            self.Ws.append(random.normal(self.rng, (layer_sizes[i-1], layer_sizes[i]))*jnp.sqrt(2.0/layer_sizes[i-1]))
            self.bs.append(jnp.zeros(layer_sizes[i]))
            
    def update(self, x, y, learning_rate):
        grads = grad(self.loss_f)(list(zip(self.Ws, self.bs)), x, y)
        for i in range(len(self.Ws)):
            self.Ws[i] = self.Ws[i] - learning_rate * grads[i][0]
            self.bs[i] = self.bs[i] - learning_rate * grads[i][1]
        
    def train(self, X, y, epochs=100, learning_rate=0.01):
        for _ in range(epochs):
            self.update(X, y, learning_rate)
            loss = self.loss_f(list(zip(self.Ws, self.bs)), X, y)
            print(f'Epoch: {_}, Loss: {loss}')
    
    def train_loader(self, loader, epochs=100, learning_rate=0.01):
        for _ in range(epochs):
            print(f'Epoch: {_}')
            loss = 0
            batches = 0
            for X, y in loader():
                self.update(X, y, learning_rate)
                params = list(zip(self.Ws, self.bs))
                loss += self.loss_f(params, X, y)
                self.Ws, self.bs = zip(*params)
                loss += self.loss_f(params, X, y)
                batches += 1
            print(f'Loss: {loss/batches}')
            
    def __call__(self, x):
        return _batched_predict(list(zip(self.Ws, self.bs)), x)

#log entropy
def log_entropy(params, x, y):
    y_hat = _batched_predict(params, x)
    return -jnp.mean(y_hat * y)

def main():
    train_images, train_labels, test_images, test_labels = datatransform.readMNIST('./../data/MNIST/')
    num_pixels = 28 * 28

    train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    
    num_labels = 10

    train_labels = one_hot(train_labels, num_labels)
    
    model = NNDense([784, 512, 512, 10], log_entropy)
    
    model.train(train_images, train_labels, epochs=10, learning_rate=0.01)
    
    test_images = model(test_images)
    test_labels_hat = jnp.argmax(test_images, axis=1)
    
    accuracy = jnp.mean(test_labels == test_labels_hat)
    print(f'Accuracy: {accuracy}')
    
    
    
def simple():

    def to_learn(x):
        # return x[0]**2 + x[1]**2 + x[2]**2
        return x[0]**2+ 2
    
    rng = random.PRNGKey(0)
    input_size = 1
    
    input_data = random.normal(rng, (10000, input_size))
    desired_output = jax.vmap(to_learn)(input_data)
    
    model = NNDense([input_size, 10, 10, 1])
    
    # model.train(input_data, desired_output, epochs=100, learning_rate=.1)
    
    batch_size = 32
    def loader():
        for i in range(0, len(input_data), batch_size):
            yield input_data[i:i+batch_size], desired_output[i:i+batch_size]
            
    model.train_loader(loader, epochs=3, learning_rate=0.1)
    
    test_data = random.normal(rng, (100, input_size))
    test_output = jax.vmap(to_learn)(test_data)
    
    test_output_hat = NNDense._batched_predict(list(zip(model.Ws, model.bs)), test_data)
    
    loss = jnp.mean((test_output - test_output_hat) ** 2)
    
    print(f'Loss: {loss}')
    n_show = 10
    print(f'Actual: \n{test_output[:n_show]}')
    print(f'Predicted: \n{test_output_hat[:n_show]}')
    
    
if __name__ == "__main__":
    main()
    
    
    