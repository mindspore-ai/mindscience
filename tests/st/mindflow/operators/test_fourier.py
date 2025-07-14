# ============================================================================
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Optimizers Test Case"""
import os
import random
import sys
from time import time as toc
import pytest
import numpy as np
from scipy.fft import dct, dst
import mindspore as ms
from mindspore import set_seed, ops
from mindflow import DFTn, IDFTn, RDFTn, IRDFTn, DCT, IDCT, DST, IDST

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# pylint: disable=wrong-import-position

from common.cell import FP32_RTOL
from common.cell.utils import compare_output

# pylint: enable=wrong-import-position

set_seed(0)
np.random.seed(0)
random.seed(0)


def gen_input(shape=(5, 6, 4, 8), rand_test=True):
    ''' Generate random or deterministic tensor for input of the tests
    '''
    a = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    if not rand_test:
        a = sum([np.arange(n).reshape([n] + [1] * j) for j, n in enumerate(shape[::-1])]) + 1j * \
            sum([np.arange(n).reshape([n] + [1] * j) for j, n in enumerate(shape[::-1])])
    ar, ai = (ms.Tensor(a.real, dtype=ms.float32), ms.Tensor(a.imag, dtype=ms.float32))
    return a, ar, ai


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['CPU', 'Ascend'])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_rdft_accuracy(device_target, mode, ndim):
    """
    Feature: Test RDFTn & IRDFTn accuracy
    Description: Input random tensor, compare the results of RDFTn and IRDFTn with numpy results
    Expectation: The output tensors should be equal within tolerance
    """
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, _ = gen_input()
    shape = a.shape

    b = np.fft.rfftn(a.real, s=a.shape[-ndim:], axes=range(-ndim, 0))
    br, bi = RDFTn(shape[-ndim:])(ar)
    cr = IRDFTn(shape[-ndim:])(br, bi)

    assert compare_output(b.real, br.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(b))
    assert compare_output(b.imag, bi.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(b))
    assert compare_output(a.real, cr.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(a))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['CPU', 'Ascend'])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_dft_accuracy(device_target, mode, ndim):
    """
    Feature: Test DFTn & IDFTn accuracy
    Description: Input random tensor, compare the results of DFTn and IDFTn with numpy results
    Expectation: The output tensors should be equal within tolerance
    """
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, ai = gen_input()
    shape = a.shape

    b = np.fft.fftn(a, s=a.shape[-ndim:], axes=range(-ndim, 0))
    br, bi = DFTn(shape[-ndim:])(ar, ai)
    cr, ci = IDFTn(shape[-ndim:])(br, bi)

    assert compare_output(b.real, br.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(b))
    assert compare_output(b.imag, bi.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(b))
    assert compare_output(a.real, cr.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(a))
    assert compare_output(a.imag, ci.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(a))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['CPU', 'Ascend'])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_dct_accuracy(device_target, mode):
    """
    Feature: Test DCT & IDCT accuracy
    Description: Input random tensor, compare the results of DCT and IDCT with numpy results
    Expectation: The output tensors should be equal within tolerance
    """
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, _ = gen_input()
    shape = a.shape

    b = dct(a.real)
    br = DCT(shape[-1:])(ar)
    cr = IDCT(shape[-1:])(br)

    assert compare_output(b.real, br.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(b))
    assert compare_output(a.real, cr.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(a))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['CPU', 'Ascend'])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_dst_accuracy(device_target, mode):
    """
    Feature: Test DST & IDST accuracy
    Description: Input random tensor, compare the results of DST and IDST with numpy results
    Expectation: The output tensors should be equal within tolerance
    """
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, _ = gen_input()
    shape = a.shape

    b = dst(a.real)
    br = DST(shape[-1:])(ar)
    cr = IDST(shape[-1:])(br)

    assert compare_output(b.real, br.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(b))
    assert compare_output(a.real, cr.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(a))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['Ascend'])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_dft_speed(device_target, mode, ndim):
    """
    Feature: Test DFTn & IDFTn speed
    Description: Input random tensor, clock the time of 10 runs of the
                gradient function containing DFT & iDFT operators
    Expectation: The average time of each run should be within 0.5s
    """
    # test dftn & idftn speed
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, ai = gen_input(shape=(64, 128, 256))
    shape = a.shape

    warmup_steps = 10
    timed_steps = 10

    dft_cell = DFTn(shape[-ndim:])
    idft_cell = IDFTn(shape[-ndim:])

    def forward_fn(xr, xi):
        br, bi = dft_cell(xr, xi)
        cr, ci = idft_cell(br, bi)
        return ops.sum(cr * cr + ci * ci)

    grad_fn = ms.value_and_grad(forward_fn, grad_position=(0, 1))

    # warmup run
    for _ in range(warmup_steps):
        _, (g1, g2) = grad_fn(ar, ai)
        ar = ar - .1 * g1
        ai = ai - .1 * g2

    # timed run
    tic = toc()
    for _ in range(timed_steps):
        _, (g1, g2) = grad_fn(ar, ai)
        ar = ar - .1 * g1
        ai = ai - .1 * g2

    assert (toc() - tic) / timed_steps < 0.5


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['CPU', 'Ascend'])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_dft_grad(device_target, mode, ndim):
    """
    Feature: Test the correctness of DFTn & IDFTn grad calculation
    Description: Input random tensor, compare the autograd results with theoretic solutions
    Expectation: The autograd results should be equal to theoretic solutions
    """
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, ai = gen_input()
    shape = a.shape

    dft_cell = DFTn(shape[-ndim:])

    def forward_fn(xr, xi):
        yr, yi = dft_cell(xr, xi)
        return ops.sum(yr * yr + yi * yi)

    grad_fn = ms.value_and_grad(forward_fn, grad_position=(0, 1))
    _, (g1, g2) = grad_fn(ar, ai)

    # analytic solution of the gradient
    b = np.fft.fftn(a, s=a.shape[-ndim:], axes=range(-ndim, 0))
    g = np.fft.ifftn(b, s=a.shape[-ndim:], axes=range(-ndim, 0)) * 2 * np.prod(a.shape[-ndim:])

    assert compare_output(g.real, g1.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(g))
    assert compare_output(g.imag, g2.numpy(), rtol=FP32_RTOL, atol=FP32_RTOL * np.linalg.norm(g))
