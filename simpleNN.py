import numpy as np

class Differentable:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def __call__(x):
        pass
    
    def backward(dLdy):
        pass
    

class ReLU(Differentable):
    def __init__(self):
        super().__init__(None, None)
    
    def __call__(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, dLdy):
        return dLdy * (self.x > 0)
    

class Sigmoid(Differentable):
    def __init__(self):
        super().__init__(None, None)
    
    def __call__(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))
    
    def backward(self, dLdy):
        out = self.__call__(self.x)
        return dLdy * out * (1 - out)
    


class NNDense:
    def __init__(self, layer_sizes, activations):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights = [[]]
        self.biases = [[]]
        self.a = [[]]
        self.z = [[]]
        self.delt = [[]]
        self.dLdw = [[]]
        self.dLdb = [[]]
        self.L = len(layer_sizes)-1
        
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]))
            self.biases.append(np.random.randn(layer_sizes[i]))
            self.dLdw.append(np.zeros((layer_sizes[i-1], layer_sizes[i])))
            self.dLdb.append(np.zeros(layer_sizes[i]))
        for i in range(len(layer_sizes)):
            self.a.append(np.zeros(layer_sizes[i]))
            self.z.append(np.zeros(layer_sizes[i]))
            self.delt.append(np.zeros(layer_sizes[i]))
            
    def forward(self, x):
        self.a[0] = x
        for l in range(1, self.L+1):
            self.z[l] = np.dot(self.a[l-1], self.weights[l]) + self.biases[l]
            self.a[l] = self.activations[l-1](self.z[l])
        return self.a[self.L]
    
    def backward(self, dLdy):
        self.delt[self.L] = dLdy
        for l in range(self.L-1, 0, -1):
            self.delt[l] = np.dot(self.delt[l+1], self.weights[l+1].T) * self.activations[l-1].backward(self.z[l])
        
        for l in range(1, self.L+1):
            self.dLdw[l] += np.outer(self.a[l-1], self.delt[l])
            self.dLdb[l] += self.delt[l]
            
    def step(self, lr):
        for l in range(1, self.L+1):
            self.weights[l] -= lr * self.dLdw[l]
            self.biases[l] -= lr * self.dLdb[l]
            
    def zero_grad(self):
        for l in range(1, self.L+1):
            self.dLdw[l] = np.zeros_like(self.dLdw[l])
            self.dLdb[l] = np.zeros_like(self.dLdb[l])
            

class MAE:
    def __init__(self):
        pass
    
    def __call__(self, y, y_pred):
        return np.mean(np.abs(y - y_pred))
    
    def backward(self, y, y_pred):
        return np.sign(y - y_pred) / y.shape[0]
    
    
def f(x):
    return x[1] + x[2]

model = NNDense([3, 5, 1], [ReLU(), Sigmoid()])

batch_size = 100
lr = 0.00001
loss = MAE()

for i in range(10000):
    model.zero_grad()
    avg_loss = 0
    for b in range(batch_size):
        x = np.random.randn(3)
        y = f(x)
        y = np.array([y])
        y_pred = model.forward(x)
        model.backward(loss.backward(y, y_pred)/batch_size)
        avg_loss += loss(y, y_pred)
    avg_loss /= batch_size
    # print(model.dLdw)
    print(avg_loss)
    model.step(lr)
    
    
    