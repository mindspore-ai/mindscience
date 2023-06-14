import pytest
import numpy as np

from mindspore import Tensor, ops, float32

from mindchemistry.e3 import BatchNorm


def test_batchnorm():
    irreps = "1x0e+2x1o"
    bn = BatchNorm(irreps, affine=False)
    v = Tensor([[0., 1., 0., 0., 0., 1., 0.]], dtype=float32)
    out = bn(v)


if __name__ == '__main__':
    test_batchnorm()
