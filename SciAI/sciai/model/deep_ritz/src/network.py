"""Network module for deep_ritz"""
import math
from abc import ABC, abstractmethod

from mindspore import nn, ops
from mindspore.common.initializer import HeUniform
from sciai.architecture import MLP


class RitzNet(nn.Cell):
    def __init__(self, layers):
        super(RitzNet, self).__init__()
        he_uniform = HeUniform(negative_slope=math.sqrt(5))
        self.mlp = MLP(layers, weight_init=he_uniform, bias_init="zeros", activation=ops.Tanh())

    def construct(self, x):
        """Network forward pass"""
        out = self.mlp(x)
        return out


class Problem(ABC):
    """Abstract class for problem definition"""
    def __init__(self, args):
        super().__init__()
        self.args = args

    @abstractmethod
    def init_net(self):
        pass

    @abstractmethod
    def train(self, net, dtype):
        pass

    @abstractmethod
    def evaluate(self, net, dtype):
        pass


def count_parameters(model):
    size = ops.Size()
    return sum(size(p) for p in model.get_parameters())
