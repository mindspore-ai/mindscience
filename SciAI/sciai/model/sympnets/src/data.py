"""data"""
import mindspore as ms
import numpy as np

from sciai.utils import amp2datatype


class Data:
    """Standard data format."""

    def __init__(self, dtype):
        self.x_train, self.y_train, self.x_test, self.y_test, self.__dtype = None, None, None, None, dtype

    @property
    def dtype(self):
        return self.__dtype

    @property
    def x_train_np(self):
        return Data.to_np(self.x_train)

    @property
    def y_train_np(self):
        return Data.to_np(self.y_train)

    @property
    def x_test_np(self):
        return Data.to_np(self.x_test)

    @property
    def y_test_np(self):
        return Data.to_np(self.y_test)

    @dtype.setter
    def dtype(self, d):
        data_type = amp2datatype(d)
        for data in ['x_train', 'y_train', 'x_test', 'y_test']:
            if isinstance(getattr(self, data), np.ndarray):
                setattr(self, data, ms.Tensor(getattr(self, data), data_type))
        self.__dtype = d

    @staticmethod
    def to_np(d):
        if isinstance(d, np.ndarray) or d is None:
            return d
        if isinstance(d, ms.Tensor):
            return d.asnumpy()
        raise ValueError()
