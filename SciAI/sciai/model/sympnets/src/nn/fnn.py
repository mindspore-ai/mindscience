"""fnn"""
from mindspore import nn
from sciai.architecture import MLP

from .module import StructureNN


class FNN(StructureNN):
    """Fully connected neural networks."""

    def __init__(self, ind, outd, layers=2, width=50, activation='relu', initializer='default', with_softmax=False):
        super(FNN, self).__init__()
        self.activation = activation
        self.initializer = initializer
        self.with_softmax = with_softmax
        self.softmax = nn.Softmax(axis=-1)
        if layers > 1:
            layers = [ind] + [width] * (layers - 2) + [outd]
            self.mlp = MLP(layers, weight_init=self.weight_init_, bias_init="zeros", activation=self.act)
        else:
            self.mlp = nn.Dense(ind, outd, weight_init=self.weight_init_, bias_init="zeros")

    def construct(self, x):
        """Network forward pass"""
        x = self.mlp(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x
