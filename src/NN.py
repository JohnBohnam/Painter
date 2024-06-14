import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit, random

import datatransform
from jax.scipy.special import logsumexp
from typing import List, Tuple

class Layer:
    # @jit
    def forward(params, x):
        raise NotImplementedError()
    
    def init_params(rng, shape):
        return jnp.array([])

class LayerMatMul(Layer):
    # @jit
    def forward(params, x):
        return jnp.dot(x, params)
    
    def init_params(rng, shape):
        input_shape, output_shape = shape
        W = random.normal(rng, (input_shape, output_shape))*jnp.sqrt(2.0/input_shape)
        return W
    
class LayerBias(Layer):
    # @jit
    def forward(params, x):
        # print('LayerBias:', x.shape, params.shape)
        return x + params
    
    def init_params(rng, shape):
        return jnp.zeros(shape)

class LReLU(Layer):
    # @jit
    def forward(params, x):
        return jax.nn.leaky_relu(x)
    
    
class LayerConv2D(Layer):
    # @jit
    def forward(params, x):
        # print('LayerConv2D:', x.shape, params.shape)
        out = jax.lax.conv(x, 
                           jnp.transpose(params, [3, 2, 0, 1]),
                           (1, 1),
                           'SAME')
        # out = x
        return out

    def init_params(rng, shape):
        # return random.normal(rng, shape)
        return jnp.zeros(shape)
    
class LayerFlatten(Layer):
    # @jit
    def forward(params, x):
        return jnp.reshape(x, (x.shape[0], -1))
    
class Layer2DReshape(Layer):
    def forward(params, x):
        # make x from vector to square matrix
        size = x.shape[1] // 16 # TODO: fix later
        res = jnp.reshape(x, (x.shape[0], 16, int(np.sqrt(size)), int(np.sqrt(size))))
        res = jnp.transpose(res, [0, 2, 3, 1])
        return res
    
    # for some reason gradient is not working if to initialize using init_params and pass the size as param
    
class Layer2DReshape1(Layer):
    def forward(params, x):
        # make x from vector to square matrix
        size = x.shape[1] // 1 # TODO: fix later
        return jnp.reshape(x, (x.shape[0], 1, int(np.sqrt(size)), int(np.sqrt(size))))
    
    
class LayerConv2DTranspose(Layer):
    def forward(params, x):
        # print('LayerConv2DTranspose:', x.shape, params.shape)    
        # kernel_rot = jnp.rot90(jnp.rot90(params, axes=(0,1)), axes=(0,1))
        # print('LayerConv2DTranspose:', x.shape, params.shape)
        out = jax.lax.conv_transpose(x, 
                                    #  jnp.transpose(params, [3, 2, 0, 1]),
                                    params,
                                     (1, 1),
                                     'SAME')
        return out
        
    def init_params(rng, shape):
        return jnp.zeros(shape)
