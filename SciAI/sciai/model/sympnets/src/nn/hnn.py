"""hnn"""
import numpy as np
from mindspore import Tensor, nn
from sciai.utils.python_utils import lazy_property

from .fnn import FNN
from .module import LossNN
from ..stormer_verlet import StormerVerlet
from ..utils import grad


class HNN(LossNN):
    """Hamiltonian neural networks."""

    def __init__(self, dim, layers=3, width=30, activation='tanh', initializer='orthogonal'):
        super(HNN, self).__init__()
        self.dim = dim
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.fnn = FNN(dim, 1, layers, width, activation, initializer)

    @property
    def third_parameter(self):
        return self.width

    def criterion(self, x0h, x1):
        x0, h = x0h[..., :-1], x0h[..., -1:]
        mid = (x0 + x1) / 2  # mid point integrator
        grad_h = grad(self.fnn, mid)
        return nn.MSELoss()((x1 - x0) / h, grad_h * self.j)

    def predict(self, x0, h, steps=1):
        n = max(int(h * 10), 1)
        solver = StormerVerlet(self.fnn, None, 10, 4, n)
        res = solver.flow(x0, h, steps)
        return res.asnumpy()

    @lazy_property
    def j(self):
        d = int(self.dim / 2)
        res = np.eye(self.dim, k=d) - np.eye(self.dim, k=-d)
        return Tensor(res, dtype=self.data_type)
