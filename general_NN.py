import numpy as np

class Differentiable:
    def __init__(self, f, df, input_shape, output_shape):
        self.f = f
        self.df = df # df is a matrix -> df[i, j] = df_i/dx_j
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_value = np.zeros(input_shape)
        self.output_value = np.zeros(output_shape)
        self.grad = np.zeros(input_shape)
        self.prev = None
        
    def __call__(self, x):
        self.input_value = x
        self.output_value = self.f(x)
        return self.output_value
        
    def backward(self, upstream_grad):
        # TODO: THIS IDEA IS FUNDAMENTALLY WRONG, TO BE FIXED
        
        # upstream_grad is the gradient of the loss with respect to the output of this layer
        # local_grad is the gradient of the output of this layer with respect to its input
        
        # upstream_grad -> [output_shape]
        # local_grad -> [input_shape, output_shape]
        local_grad = self.df(self.input_value)
        # print("Upstream grad: ", upstream_grad)
        # print("Local grad: ", local_grad)
        
        self.grad += local_grad @ upstream_grad
        # print("Grad: ", self.grad)
        
        if self.prev is not None:
            self.prev.backward(self.grad)
        
    def zero_grad(self):
        self.grad = np.zeros(self.input_shape)
        if self.prev is not None:
            self.prev.zero_grad()



class LayerMatmul(Differentiable):
    def __init__(self, input_shape, output_shape):
        super().__init__(self.f, self.df, input_shape, output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape
        # self.W = np.random.randn(output_shape, input_shape)
        self.W = np.zeros([output_shape, input_shape])+1
        self.grad = np.zeros(self.W.shape)
        
    def f(self, x):
        return self.W @ x
    
    def df(self, x):
        return self.W.T
        

class LayerBias(Differentiable):
    def __init__(self, input_shape):
        super().__init__(self.f, self.df, input_shape, input_shape)
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.b = np.random.randn(input_shape)
        self.grad = np.zeros(self.b.shape)
        
    def f(self, x):
        return x + self.b
    
    def df(self, x):
        return np.ones(self.b.shape)
    
        
class LayerDense(Differentiable):
    def __init__(self, input_shape, output_shape):
        super().__init__(self.f, self.df, input_shape, output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.matmul = LayerMatmul(input_shape, output_shape)
        self.bias = LayerBias(output_shape)
        
    def f(self, x):
        return self.bias(self.matmul(x))
    
    def df(self, x):
        pass
    
    def backward(self, x):
        return self.bias.backward(self.matmul.backward(x))
    
    def zero_grad(self):
        self.matmul.zero_grad()
        self.bias.zero_grad()
        

class ReLU(Differentiable):
    def __init__(self, input_shape, output_shape):
        super().__init__(self.f, self.df, input_shape, output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def f(self, x):
        return np.maximum(x, 0)
    
    def df(self, x):
        return (x > 0).astype(int)  


def f(x, y):
    return x * 3 + 2 * y

input_size = 2
output_size = 1

dense = LayerMatmul(input_size, output_size)
print(dense.W)

lr = 0.1

batch_size = 50

for i in range(100):
    dense.zero_grad()
    loss = 0
    for b in range(batch_size):
        x = np.random.randn(2)
        y = f(*x)
        y_pred = dense(x)
        loss += (y_pred - y)**2
        dense.backward(2*(y_pred - y)/batch_size)
        
    print(f"grad: {dense.grad}")
    loss/=batch_size
    print(f"loss: {loss}")
    dense.W -= lr * dense.grad
    
    print(dense.W)
