import mindspore as ms
import numpy as np
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import XavierNormal

from sciai.architecture import MLP
from sciai.operators import grad


class ODENN(nn.Cell):
    def __init__(self, fnn):
        super(ODENN, self).__init__()
        self.fnn = fnn
        self.grad = grad(self.fnn, output_index=0, input_index=0)

    def construct(self, t):
        """Network forward pass"""
        u_t = self.grad(t)
        t_new = ops.mul(t, 0.5 * np.pi)
        f = ops.sub(u_t, ops.add(ops.mul(ops.cos(t_new), 0.5 * np.pi), 1))

        return f


class FNN(nn.Cell):
    def __init__(self, layers, x_min, x_max):
        super(FNN, self).__init__()
        self.size = layers
        self.x_min = Tensor(x_min, dtype=ms.float32)
        self.x_max = Tensor(x_max, dtype=ms.float32)
        self.mlp = MLP(layers, weight_init=XavierNormal())

    def construct(self, x):
        """Network forward pass"""
        a = ops.sub(x, self.x_min) / ops.sub(self.x_max, self.x_min)
        a = ops.mul(a, 2.0)
        a = ops.sub(a, 1.0)
        y = self.mlp(a)

        return y
