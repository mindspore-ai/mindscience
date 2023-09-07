"""process for sympnets"""
import numpy as np
from mindspore import ops, nn

from sciai import operators


class OpsUtil:
    """utilities for ops"""
    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self):
        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()
        self.log_softmax = ops.LogSoftmax(axis=-1)
        self.cat = ops.Concat(axis=-1)
        self.ce = nn.CrossEntropyLoss()


def cross_entropy_loss(y_pred, y_label):
    if y_pred.size() == y_label.size():
        return OpsUtil().mean(-OpsUtil().sum(OpsUtil().log_softmax(y_pred) * y_label, -1))
    return OpsUtil().ce(y_pred, y_label)


def grad(net, x, keepdim=False):
    """
    y: [n, ny] or [ny]
    x: [n, nx] or [nx]
    Return dy/dx ([n, ny, nx] or [ny, nx]).
    """
    y = net(x)
    n = y.size(0) if len(y.size()) == 2 else 1
    ny = y.size(-1)
    nx = x.size(-1)
    dy = []
    grad_net = operators.grad(net)
    grad_res = grad_net(x)
    for i in range(ny):
        dy.append(grad_res[i])
    shape = np.array([n, ny])[2 - len(net.size()):]
    shape = list(shape) if keepdim else list(shape[shape > 1])
    return OpsUtil().cat(dy).view(shape + [nx])
