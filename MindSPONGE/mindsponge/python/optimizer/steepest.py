"""
Optimizer used to get the minimum value of a given function.
"""
import mindspore as ms
from mindspore import nn, Parameter, Tensor
from mindspore import numpy as msnp


class SteepestDescent(nn.Optimizer):
    """
    The steepest descent (gradient descent) optimizer with growing learning rate.

    Args:
        crd(tuple):             Usually a tuple of parameters is given and the first element is coordinates.
        learning_rate(float):   A factor of each optimize step size.
        factor(float):          A growing factor of learning rate.
        nonh_mask(Tensor):      The mask of atoms which are not Hydrogen.
        max_shift(float):       The max step size each atom can move.

    Returns:
        float, the first element of parameters.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, crd, learning_rate=1e-03, factor=1.001, nonh_mask=None, max_shift=1.0):
        super(SteepestDescent, self).__init__(learning_rate, crd)
        self.crd = crd[0]
        self.learning_rate = Parameter(Tensor(learning_rate, ms.float32))
        self.factor = Parameter(Tensor(factor, ms.float32))
        if nonh_mask is not None:
            self.nonh_mask = nonh_mask
        else:
            self.nonh_mask = msnp.ones((1, self.crd.shape[-2], 1))
        self.max_shift = Parameter(Tensor(max_shift, ms.float32))

    def construct(self, gradients):
        shift = self.learning_rate*gradients[0]*self.nonh_mask
        shift = msnp.where(shift > self.max_shift, self.max_shift, shift)
        shift = msnp.where(shift < -self.max_shift, -self.max_shift, shift)
        self.crd -= shift
        self.learning_rate *= self.factor
        return self.crd
