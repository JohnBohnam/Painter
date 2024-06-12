import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, random

import datatransform
from jax.scipy.special import logsumexp

class Layer:
    def forward(params, x):
        raise NotImplementedError()
    
    def init_params(rng, shape):
        return jnp.array([])

class LayerMatMul(Layer):
    def forward(params, x):
        return jnp.dot(x, params)
    
    def init_params(rng, shape):
        input_shape, output_shape = shape
        W = random.normal(rng, (input_shape, output_shape))*jnp.sqrt(2.0/input_shape)
        return W
    
class LayerBias(Layer):
    def forward(params, x):
        return x + params
    
    def init_params(rng, shape):
        return jnp.zeros(shape)

class LReLU(Layer):
    def forward(params, x):
        return jax.nn.leaky_relu(x)
    
    
class LayerConv2D(Layer):
    def forward(params, x):
        # print(f"kernel shape: {params.shape}")
        out = jax.lax.conv(jnp.transpose(x,[0,3,1,2]),    # lhs = NCHW image tensor
                jnp.transpose(params,[3,2,0,1]), # rhs = OIHW conv kernel tensor
                (1, 1),  # window strides
                'SAME') # padding mode
        return out
    
    def init_params(rng, shape):
        return random.normal(rng, shape)
    
class LayerFlatten(Layer):
    def forward(params, x):
        return jnp.reshape(x, (x.shape[0], -1))
