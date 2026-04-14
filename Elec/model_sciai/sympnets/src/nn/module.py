"""module"""
from abc import abstractmethod

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import HeNormal, HeUniform, XavierUniform, XavierNormal, Orthogonal


class Module(nn.Cell):
    """Standard module format.
    """

    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        self.__dtype = None
        self.__amp_level = None

        self.activation_dict = {'sigmoid': ops.Sigmoid(), 'relu': ops.ReLU(), 'tanh': ops.Tanh(), 'elu': ops.Elu()}
        self.initializer_dict = {'He normal': HeNormal(), 'He uniform': HeUniform(), 'Glorot normal': XavierNormal(),
                                 'Glorot uniform': XavierUniform(), 'orthogonal': Orthogonal()}

    @property
    def dtype(self):
        return self.__dtype

    @property
    def data_type(self):
        if self.__dtype == 'float':
            return ms.float32
        if self.__dtype == 'double':
            return ms.float64
        raise NotImplementedError()

    @property
    def act(self):
        if self.activation in self.activation_dict:
            return self.activation_dict.get(self.activation)
        raise NotImplementedError()

    @property
    def weight_init_(self):
        """weight init"""
        if self.initializer in self.initializer_dict:
            return self.initializer_dict.get(self.initializer)
        if self.initializer == 'default':
            if self.activation == 'relu':
                return HeNormal()
            if self.activation == 'tanh':
                return Orthogonal()
            raise NotImplementedError()
        raise NotImplementedError()

    @dtype.setter
    def dtype(self, d):
        if d in ('O3', "O2", "O1"):
            self.to_float(ms.float16)
        elif d == 'O0':
            self.to_float(ms.float32)
        else:
            raise ValueError()
        self.__dtype = d


class StructureNN(Module):
    """Structure-oriented neural network used as a general map based on designing architecture.
    """

    def predict(self, x):
        """predict"""
        return self(x)


class LossNN(Module):
    """Loss-oriented neural network used as an algorithm based on designing loss.
    """

    def construct(self, x):
        """Network forward pass"""
        return x

    @abstractmethod
    def criterion(self, x0h, x1):
        pass
