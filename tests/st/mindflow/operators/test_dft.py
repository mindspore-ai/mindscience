# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test mindflow dft"""

import torch
import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
from mindflow.cell.neural_operators.dft import dft1, dft2, idft1, idft2


def dft_1d_torch(x, dim=-1):
    x = torch.Tensor(x)

    x_re_im = torch.fft.fft(x, dim=dim, norm="ortho")
    x_re, x_im = x_re_im.real, x_re_im.imag
    return x_re.numpy(), x_im.numpy()


def dft_2d_torch(x, dim=-1):
    x = torch.Tensor(x)

    x_re_im = torch.fft.rfft2(x, dim=dim, norm="ortho")
    x_re, x_im = x_re_im.real, x_re_im.imag
    return x_re.numpy(), x_im.numpy()


def idft_1d_torch(x_re, x_im, dim=-1):
    x = torch.stack([torch.Tensor(x_re), torch.Tensor(x_im)], dim=-1)
    x = torch.view_as_complex(x)
    x = torch.fft.ifft(x, norm="ortho", dim=dim)
    return x.numpy()


def idft_2d_torch(x_re, x_im, dim=-1):
    x = torch.stack([torch.Tensor(x_re), torch.Tensor(x_im)], dim=-1)
    x = torch.view_as_complex(x)
    x = torch.fft.irfft2(x, norm="ortho", dim=dim)
    return x.numpy()


def dft_1d_ms(x, shape, mode, dim=(-1,)):
    x = ms.Tensor(x)
    x_re = x
    x_im = ops.zeros_like(x_re)
    dft1_cell = dft1(shape=shape, modes=mode, dim=dim)
    x_ft_re, x_ft_im = dft1_cell((x_re, x_im))
    return x_ft_re.asnumpy(), x_ft_im.asnumpy()


def dft_2d_ms(x, shape, mode, dim=(-1,)):
    x = ms.Tensor(x)
    x_re = x
    x_im = ops.zeros_like(x_re)
    dft2_cell = dft2(shape=shape, modes=mode, dim=dim)
    x_ft_re, x_ft_im = dft2_cell((x_re, x_im))
    return x_ft_re.asnumpy(), x_ft_im.asnumpy()


def idft_1d_ms(x_re, x_im, shape, mode, dim=(-1)):
    x_re = ms.Tensor(x_re)
    x_im = ms.Tensor(x_im)
    idft1_cell = idft1(shape=shape, modes=mode, dim=dim)
    x_ms, _ = idft1_cell((x_re, x_im))
    return x_ms.asnumpy()


def idft_2d_ms(x_re, x_im, shape, mode, dim=(-1)):
    x_re = ms.Tensor(x_re)
    x_im = ms.Tensor(x_im)
    idft2_cell = idft2(shape=shape, modes=mode, dim=dim)
    x_ms, _ = idft2_cell((x_re, x_im))
    return x_ms.asnumpy()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dft1d():
    """
    Feature: Test dft1d in platform gpu and ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    Torch problem, need to adaptive 910B
    """
    x = np.random.randn(1, 6, 8, 2)
    x_re_torch1d, x_im_torch1d = dft_1d_torch(x, dim=-2)
    x_re_ms1d, x_im_ms1d = dft_1d_ms(x, shape=(8,), mode=5, dim=(-2,))

    x_torch1d = idft_1d_torch(x_re_torch1d, x_im_torch1d, dim=-2)
    x_ms1d = idft_1d_ms(x_re_ms1d, x_im_ms1d, shape=(8,), mode=5, dim=(-2,))

    assert np.sum(x_torch1d - x_ms1d) < 0.001


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dft2d():
    """
    Feature: Test dft2d in platform gpu and ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    Torch problem, need to adaptive 910B
    """
    x = np.random.randn(1, 6, 8, 2)
    x_re_torch2d, x_im_torch2d = dft_2d_torch(x, dim=(-3, -2))
    x_re_ms2d, x_im_ms2d = dft_2d_ms(x, shape=(6, 8), mode=(3, 5), dim=(-3, -2))

    x_torch2d = idft_2d_torch(x_re_torch2d, x_im_torch2d, dim=(-3, -2))
    x_ms2d = idft_2d_ms(x_re_ms2d, x_im_ms2d, shape=(6, 8), mode=(3, 5), dim=(-3, -2))

    assert np.sum(x_torch2d - x_ms2d) < 0.001
