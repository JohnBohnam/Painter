from NeuralNet import NeuralNet
from jax import grad

class AutoEncoder:
    def _init_locals(self):
        encoder_params = self.encoder.params
        decoder_params = self.decoder.params
        
        encoder_layers = self.encoder.layers
        decoder_layers = self.decoder.layers
        
        encoder_params, decoder_params = self.params
        
        self.encoder.params = encoder_params
        self.decoder.params = decoder_params
        
        self.encoder.layers = encoder_layers
        self.decoder.layers = decoder_layers
        
        self.encoder._init_locals()
        self.decoder._init_locals()
        
        
        def forward(params, x):
            encoder_params, decoder_params = params
            encoded = self.encoder.forward(encoder_params, x)
            decoded = self.decoder.forward(decoder_params, encoded)
            return decoded

        def loss(params, x, y):
            y_hat = forward(params, x)
            return self.loss_f(y_hat, y)

        self.forward = forward
        self.loss = loss
    
    def __init__(self, encoder_layers, encoder_layer_shapes, decoder_layers, decoder_layer_shapes, loss_f):
        self.encoder = NeuralNet(encoder_layers, encoder_layer_shapes, loss_f)
        self.decoder = NeuralNet(decoder_layers, decoder_layer_shapes, loss_f)
        self.loss_f = loss_f
        self.params = (self.encoder.params, self.decoder.params)

        self._init_locals()
        print(f"AutoEncoder initialized with {len(self.params[0])} encoder layers and {len(self.params[1])} decoder layers")

    def __call__(self, x):
        return self.forward(self.params, x)

    def update(self, x, y, learning_rate):
        grads = grad(self.loss)(self.params, x, y)
        enc_grads, dec_grads = grads

        self.encoder.updateWithGrad(enc_grads, learning_rate)
        self.decoder.updateWithGrad(dec_grads, learning_rate)
        self.params = (self.encoder.params, self.decoder.params)
        # print(self.params)

    def train(self, X, y, epochs=100, learning_rate=0.01):
        # implement mini-batch training
        batch_size = 512
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                x_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.update(x_batch, y_batch, learning_rate)
            loss = self.loss(self.params, X, y)
            print(f'Epoch: {epoch}, Loss: {loss}')

    def save(self, path):
        # save model into the file
        import pickle
        with open(path, 'wb') as f:
            pickle.dump((self.params, self.encoder.layers, self.decoder.layers, self.loss_f), f)
            

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.params, self.encoder.layers, self.decoder.layers, self.loss_f = pickle.load(f)
            self._init_locals()