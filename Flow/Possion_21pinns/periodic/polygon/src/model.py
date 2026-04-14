"""Define networks."""
import math

import mindspore as ms
from mindspore import nn, ops
from mindspore import numpy as ms_np
from mindspore.common import Parameter
from mindspore.common.initializer import initializer, HeUniform, Normal


class ExpDeacy(nn.Cell):
    """Exponentially deacy activation function."""
    def __init__(self):
        super(ExpDeacy, self).__init__()
        self.factor = ms.Tensor(-1/(2*math.e))
        self.exp = ops.Exp()
        self.square = ops.Square()

    def construct(self, x):
        return x*self.exp(self.factor*self.square(x))


class LinearWN(nn.Cell):
    """A linear layer with weight normalization."""
    def __init__(self, in_features, out_features,
                 weight_init='normal', g_init='normal', bias_init='zeros'):
        super(LinearWN, self).__init__()
        self.weight \
            = Parameter(initializer(weight_init, [in_features, out_features]), name='weight')
        self.sqrt_g = Parameter(initializer(g_init, out_features), name='sqrt_g')
        self.bias = Parameter(initializer(bias_init, out_features), name='bias')

    def construct(self, x):
        weight = self.weight
        weight = weight/ms_np.sqrt(ms_np.mean(weight*weight, axis=-1, keepdims=True))
        weight = self.sqrt_g*self.sqrt_g*weight
        return ms_np.matmul(x, weight) + self.bias


def create_model(input_size, base_neurons):
    """Create a network."""
    return nn.SequentialCell(
        nn.Dense(input_size, 8*base_neurons, HeUniform(), activation=ExpDeacy()),
        nn.Dense(8*base_neurons, 4*base_neurons, HeUniform(), activation=ExpDeacy()),
        nn.Dense(4*base_neurons, 2*base_neurons, HeUniform(), activation=ExpDeacy()),
        nn.Dense(2*base_neurons, base_neurons, HeUniform(), activation=ExpDeacy()),
        LinearWN(base_neurons, 1, HeUniform(), Normal(sigma=.1))
    )
