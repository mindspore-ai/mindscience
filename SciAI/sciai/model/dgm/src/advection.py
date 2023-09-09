"""advection"""
import mindspore as ms
from mindspore import ops
import numpy as np


class Advection:
    """Advection case"""
    def __init__(self, net):
        self.net = net
        self.grad = ops.GradOperation()
        self.grad_net = self.grad(self.net)

    @staticmethod
    def exact_solution(x: ms.Tensor):
        return ops.sin(2 * np.pi * x) * ops.cos(4 * np.pi * x) + 1

    def criterion(self, x, x_initial):
        dx = self.grad_net(x)
        domain = (dx - 2 * np.pi * ops.cos(2 * np.pi * x) * ops.cos(4 * np.pi * x) +
                  4 * np.pi * ops.sin(4 * np.pi * x) * ops.sin(2 * np.pi * x)) ** 2
        ic = (self.net(x_initial) - ms.Tensor([[1]])) ** 2
        return domain, ic
