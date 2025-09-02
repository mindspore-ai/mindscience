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
import random

import numpy as np
import mindspore as ms
import pytest

# pylint: disable=ungrouped-imports
from mindspore import set_seed, ops
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics, ExportType

from mindscience.sciops import DFTn, IDFTn, RDFTn, IRDFTn, asd_fftn, asd_ifftn, asd_rfftn, asd_irfftn

set_seed(0)
np.random.seed(0)
random.seed(0)
FP32_RTOL = 1e-3

def loss_func_c(yr, yi):
    return ops.sum(yr * yr + 2 * yi * yi)

def loss_grad_np_c(y):
    return 2 * y.real + 4j * y.imag

def loss_func_r(yr):
    return ops.sum(yr * yr * yr)

def loss_grad_np_r(y):
    return 3 * y * y

def forwad_fn_c2c(xr, xi, dim, dft_cell):
    if dim is not None:
        br, bi = dft_cell(xr, xi, ndim=dim)
    else:
        br, bi = dft_cell(xr, xi)
    return loss_func_c(br, bi)

def forwad_fn_r2c(xr, dim, dft_cell):
    if dim is not None:
        br, bi = dft_cell(xr, ndim=dim)
    else:
        br, bi = dft_cell(xr)
    return loss_func_c(br, bi)

def forwad_fn_c2r(xr, xi, dim, dft_cell):
    if dim is not None:
        br = dft_cell(xr, xi, ndim=dim)
    else:
        br = dft_cell(xr, xi)
    return loss_func_r(br)

def gen_input(shape=(2, 16, 16), rand_test=True):
    ''' Generate random or deterministic tensor for input of the tests
    '''
    a = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    if not rand_test:
        a = sum([np.arange(n).reshape([n] + [1] * j) for j, n in enumerate(shape[::-1])]) + 1j * \
            sum([np.arange(n).reshape([n] + [1] * j) for j, n in enumerate(shape[::-1])])
    ar, ai = (ms.Tensor(a.real, dtype=ms.float32), ms.Tensor(a.imag, dtype=ms.float32))
    return a, ar, ai

def cal_error(name, ar, ai, br, bi):
    '''
    ar, ai, br, bi are all numpy arrays, calculate the max absolute error, max relative error, and mean relative error
    '''
    print(f"{name} ar.shape: ", ar.shape)
    print(f"{name} br.shape: ", br.shape)
    if ai is not None and bi is not None:
        print(f"{name} ai.shape: ", ai.shape)
        print(f"{name} bi.shape: ", bi.shape)
    abs_error_real = np.abs(ar - br)
    rel_error_real = abs_error_real / (np.abs(ar) + 1e-10)
    if ai is not None and bi is not None:
        abs_error_imag = np.abs(ai - bi)
        rel_error_imag = abs_error_imag / (np.abs(ai) + 1e-10)
        max_abs_error = max(np.max(abs_error_real), np.max(abs_error_imag))
        max_rel_error = max(np.max(rel_error_real), np.max(rel_error_imag))
        mean_rel_error = (np.mean(rel_error_real) + np.mean(rel_error_imag)) / 2
    else:
        abs_error_imag = None
        rel_error_imag = None
        max_abs_error = np.max(abs_error_real)
        max_rel_error = np.max(rel_error_real)
        mean_rel_error = np.mean(rel_error_real)

    print(f"{name} max_abs_error: ", max_abs_error)
    print(f"{name} max_rel_error: ", max_rel_error)
    print(f"{name} mean_rel_error: ", mean_rel_error)

    return max_abs_error, max_rel_error, mean_rel_error


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['Ascend'])
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ndim', [1, 2])
def test_asd_fft_accuracy(device_target, mode, ndim):
    """
    Feature: Test ASD FFT & IFFT accuracy
    Description: Input random tensor, compare the results of ASD FFT and IFFT with numpy results
    Expectation: The output tensors should be equal within tolerance
    """
    print(f"test_asd_fft_accuracy, ndim: {ndim}")
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, ai = gen_input()
    shape = a.shape

    b = np.fft.fftn(a, s=a.shape[-ndim:], axes=range(-ndim, 0))
    # mindflow DFTn, real and imag are both mindspore tensors
    br, bi = DFTn(shape[-ndim:])(ar, ai)
    cr, ci = IDFTn(shape[-ndim:])(br, bi)

    # ASD FFT, real and imag are both mindspore tensors
    ms_br, ms_bi = asd_fftn(ar, ai, ndim=ndim)
    ms_ar, ms_ai = asd_ifftn(ms_br, ms_bi, ndim=ndim)

    # mindflow dft is just used for reference
    max_abs_error, max_rel_error, mean_rel_error = cal_error(
        "numpy-vs-mindflow-dft", b.real, b.imag, br.asnumpy(), bi.asnumpy())
    max_abs_error, max_rel_error, mean_rel_error = cal_error(
        "numpy-vs-ms-dft", b.real, b.imag, ms_br.asnumpy(), ms_bi.asnumpy())
    assert max_abs_error < FP32_RTOL
    assert max_rel_error < FP32_RTOL
    assert mean_rel_error < FP32_RTOL
    # mindflow idft is just used for reference
    max_abs_error, max_rel_error, mean_rel_error = cal_error(
        "numpy-vs-mindflow-idft", a.real, a.imag, cr.asnumpy(), ci.asnumpy())
    max_abs_error, max_rel_error, mean_rel_error = cal_error(
        "numpy-vs-ms-idft", a.real, a.imag, ms_ar.asnumpy(), ms_ai.asnumpy())
    assert max_abs_error < FP32_RTOL
    assert max_rel_error < FP32_RTOL
    assert mean_rel_error < FP32_RTOL


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['Ascend'])
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ndim', [1, 2])
def test_asd_rfft_accuracy(device_target, mode, ndim):
    """
    Feature: Test ASD RFFT & IRFFT accuracy
    Description: Input random tensor, compare the results of ASD RFFT and IRFFT with numpy results
    Expectation: The output tensors should be equal within tolerance
    """
    print(f"test_asd_rfft_accuracy, ndim: {ndim}")
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, _ = gen_input()
    shape = a.shape

    b = np.fft.rfftn(a.real, s=a.shape[-ndim:], axes=range(-ndim, 0))
    br, bi = RDFTn(shape[-ndim:])(ar)
    cr = IRDFTn(shape[-ndim:])(br, bi)

    ms_br, ms_bi = asd_rfftn(ar, ndim=ndim)
    ms_ar = asd_irfftn(ms_br, ms_bi, ndim=ndim)

    ms_ar_n = asd_irfftn(ms_br, ms_bi, n=ar.shape[-1] + 1, ndim=ndim)
    np_shape = list(a.shape[-ndim:])
    np_shape[-1] = np_shape[-1] + 1
    np_ar_n = np.fft.irfftn(b, s=np_shape, axes=range(-ndim, 0))
    max_abs_error, max_rel_error, mean_rel_error = cal_error(
        "numpy-vs-ms-irdft-n", np_ar_n, None, ms_ar_n.asnumpy(), None)
    assert max_abs_error < FP32_RTOL
    assert max_rel_error < FP32_RTOL
    assert mean_rel_error < FP32_RTOL

    max_abs_error, max_rel_error, mean_rel_error = cal_error(
        "numpy-vs-mindflow-rdft", b.real, b.imag, br.asnumpy(), bi.asnumpy())
    max_abs_error, max_rel_error, mean_rel_error = cal_error(
        "numpy-vs-ms-rdft", b.real, b.imag, ms_br.asnumpy(), ms_bi.asnumpy())
    assert max_abs_error < FP32_RTOL
    assert max_rel_error < FP32_RTOL
    assert mean_rel_error < FP32_RTOL
    max_abs_error, max_rel_error, mean_rel_error = cal_error(
        "numpy-vs-mindflow-irdft", a.real, a.imag, cr.asnumpy(), None)
    max_abs_error, max_rel_error, mean_rel_error = cal_error("numpy-vs-ms-irdft", a.real, a.imag, ms_ar.asnumpy(), None)
    assert max_abs_error < FP32_RTOL
    assert max_rel_error < FP32_RTOL
    assert mean_rel_error < FP32_RTOL

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['Ascend'])
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('cell', ["c2c_fwd", "c2c_inv"])
# pylint: disable=unused-variable
def test_asd_fft_grad_accuracy(device_target, mode, ndim, cell):
    """
    Feature: Test ASD FFT & IFFT grad accuracy
    Description: Input random tensor, compare the results of ASD FFT and IFFT with numpy results
    Expectation: The output tensors should be equal within tolerance
    """
    print(f"test_asd_fft_grad_accuracy, ndim: {ndim}, cell: {cell}")
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, ai = gen_input()
    shape = a.shape
    scale = np.prod(a.shape[-ndim:])
    grad_fn = ms.value_and_grad(forwad_fn_c2c, grad_position=(0, 1))

    if cell == "c2c_fwd":
        asd_result, (asd_ar_g, asd_ai_g) = grad_fn(ar, ai, ndim, asd_fftn)
        ms_result, (ms_ar_g, ms_ai_g) = grad_fn(ar, ai, None, DFTn(shape[-ndim:]))
        np_result = np.fft.fftn(a, s=a.shape[-ndim:], axes=range(-ndim, 0))
        c = loss_grad_np_c(np_result)
        np_grad = scale * np.fft.ifftn(c, s=a.shape[-ndim:], axes=range(-ndim, 0))
    elif cell == "c2c_inv":
        asd_result, (asd_ar_g, asd_ai_g) = grad_fn(ar, ai, ndim, asd_ifftn)
        ms_result, (ms_ar_g, ms_ai_g) = grad_fn(ar, ai, None, IDFTn(shape[-ndim:]))
        np_result = np.fft.ifftn(a, s=a.shape[-ndim:], axes=range(-ndim, 0))
        c = loss_grad_np_c(np_result)
        np_grad = (1.0 / scale) * np.fft.fftn(c, s=a.shape[-ndim:], axes=range(-ndim, 0))
    else:
        raise ValueError(f"fft: Unsupported cell: {cell}, only support c2c_fwd and c2c_inv")

    abs_error, rel_error, mean_error = cal_error(
        "fft-numpy-vs-mindflow-backward", np_grad.real, np_grad.imag, ms_ar_g.asnumpy(), ms_ai_g.asnumpy())
    abs_error, rel_error, mean_error = cal_error(
        "fft-numpy-vs-asdfft-backward", np_grad.real, np_grad.imag, asd_ar_g.asnumpy(), asd_ai_g.asnumpy())
    assert abs_error < FP32_RTOL
    assert rel_error < FP32_RTOL
    assert mean_error < FP32_RTOL

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['Ascend'])
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('cell', ["r2c_fwd", "c2r_inv"])
# pylint: disable=unused-variable
def test_asd_rfft_grad_accuracy(device_target, mode, ndim, cell):
    """
    Feature: Test ASD RFFT & IRFFT grad accuracy
    Description: Input random tensor, compare the results of ASD RFFT and IRFFT with numpy results
    Expectation: The output tensors should be equal within tolerance
    """
    print(f"test_asd_rfft_grad_accuracy, ndim: {ndim}, cell: {cell}")
    ms.set_context(device_target=device_target, mode=mode)
    a, ar, ai = gen_input()
    shape = a.shape
    scale = np.prod(a.shape[-ndim:])
    grad_fn_r2c = ms.value_and_grad(forwad_fn_r2c, grad_position=(0))
    grad_fn_c2r = ms.value_and_grad(forwad_fn_c2r, grad_position=(0, 1))

    n = shape[-1]
    m = n // 2 + 1
    k = (n+1) // 2
    alf = np.ones(m)
    alf[1:k] += 1


    if cell == "r2c_fwd":
        asd_result, asd_ar_g = grad_fn_r2c(ar, ndim, asd_rfftn)
        ms_result, ms_ar_g = grad_fn_r2c(ar, None, RDFTn(shape[-ndim:]))
        np_result = np.fft.rfftn(a.real, s=a.shape[-ndim:], axes=range(-ndim, 0))
        c = loss_grad_np_c(np_result) / alf
        np_grad = scale * np.fft.irfftn(c, s=a.shape[-ndim:], axes=range(-ndim, 0))
        abs_error, rel_error, mean_error = cal_error(
            "rfft-numpy-vs-mindflow-backward", np_grad, None, ms_ar_g.asnumpy(), None)
        abs_error, rel_error, mean_error = cal_error(
            "rfft-numpy-vs-asdfft-backward", np_grad, None, asd_ar_g.asnumpy(), None)

    elif cell == "c2r_inv":
        ar_input, ai_input = asd_rfftn(ar, ndim=ndim)
        asd_result, (asd_ar_g, asd_ai_g) = grad_fn_c2r(ar_input, ai_input, ndim, asd_irfftn)
        # mindflow IRDFT
        ms_result, (ms_ar_g, ms_ai_g) = grad_fn_c2r(ar_input, ai_input, None, IRDFTn(ar.shape[-ndim:]))
        make_complex = ops.Complex()
        ms_c = make_complex(ar_input, ai_input).asnumpy()
        np_result = np.fft.irfftn(ms_c, s=a.shape[-ndim:], axes=range(-ndim, 0))
        c = loss_grad_np_r(np_result)
        n = (ms_c.shape[-1] - 1) * 2
        m = n // 2 + 1
        k = (n+1) // 2
        alf = np.ones(m)
        alf[1:k] += 1
        np_grad = (1.0 / scale) * alf * np.fft.rfftn(c, s=a.shape[-ndim:], axes=range(-ndim, 0))
        abs_error, rel_error, mean_error = cal_error(
            "rfft-numpy-vs-mindflow-backward", np_grad.real, np_grad.imag, ms_ar_g.asnumpy(), ms_ai_g.asnumpy())
        abs_error, rel_error, mean_error = cal_error(
            "rfft-numpy-vs-asdfft-backward", np_grad.real, np_grad.imag, asd_ar_g.asnumpy(), asd_ai_g.asnumpy())
        assert abs_error < FP32_RTOL
        assert rel_error < FP32_RTOL
        assert mean_error < FP32_RTOL
    else:
        raise ValueError(f"rfft: Unsupported cell: {cell}, only support r2c_fwd and c2r_inv")

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('device_target', ['Ascend'])
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('alg', ["DFT", "IDFT", "RDFT", "IRDFT", "FFT", "IFFT", "RFFT", "IRFFT"])
@pytest.mark.parametrize('dshape', [(20, 512, 512), (20, 1024, 1024), (20, 2048, 2048)])
def test_fft_speed(device_target, mode, ndim, alg, dshape):
    """
    Feature: Test ASD FFT & RFFT speed
    Description: Input random tensor, compare the results of ASD FFT and RFFT with numpy results
    Expectation: The output tensors should be equal within tolerance
    """
    print(f"test_fft_speed, ndim: {ndim}, alg: {alg}, dshape: {dshape}")
    ms.set_context(device_target=device_target, mode=mode)

    experimental_config = ms.profiler._ExperimentalConfig( # pylint: disable=protected-access
        profiler_level=ProfilerLevel.Level1,
        aic_metrics=AicoreMetrics.AiCoreNone,
        l2_cache=False,
        mstx=False,
        data_simplification=False,
        export_type=[ExportType.Text])
    a, ar, ai = gen_input(shape=dshape)
    shape = a.shape
    br, bi = DFTn(shape[-ndim:])(ar, ai)

    prof_file_path = f"./data/fft_speed/{alg}_{ndim}_{dshape[0]}_{dshape[1]}_{dshape[2]}"

    with ms.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                             schedule=ms.profiler.schedule(wait=0, warmup=4, active=4, repeat=1, skip_first=0),
                             on_trace_ready=ms.profiler.tensorboard_trace_handler(prof_file_path),
                             profile_memory=True,
                             with_stack=True,
                             record_shapes=True,
                             experimental_config=experimental_config) as prof:
        for _ in range(10):
            if alg == "DFT":
                br, bi = DFTn(shape[-ndim:])(ar, ai)
            elif alg == "IDFT":
                br, bi = IDFTn(shape[-ndim:])(br, bi)
            elif alg == "RDFT":
                br = RDFTn(shape[-ndim:])(ar)
            elif alg == "IRDFT":
                ar = IRDFTn(shape[-ndim:])(br, bi)

            elif alg == "FFT":
                br, bi = asd_fftn(ar, ai, ndim=ndim)
            elif alg == "IFFT":
                ar, ai = asd_ifftn(br, bi, ndim=ndim)
            elif alg == "RFFT":
                br, bi = asd_rfftn(ar, ndim=ndim)
            elif alg == "IRFFT":
                ar = asd_irfftn(br, bi, ndim=ndim)

            prof.step()


if __name__ == "__main__":
    test_dft_accuracy(device_target='Ascend', mode=ms.PYNATIVE_MODE, ndim=1)
    test_dft_accuracy(device_target='Ascend', mode=ms.PYNATIVE_MODE, ndim=2)
    test_rdft_accuracy(device_target='Ascend', mode=ms.PYNATIVE_MODE, ndim=1)
    test_rdft_accuracy(device_target='Ascend', mode=ms.PYNATIVE_MODE, ndim=2)
    test_dft_accuracy_with_grad(device_target='Ascend', mode=ms.PYNATIVE_MODE, ndim=1)
    test_dft_accuracy_with_grad(device_target='Ascend', mode=ms.PYNATIVE_MODE, ndim=2)
    test_rdft_accuracy_with_grad(device_target='Ascend', mode=ms.PYNATIVE_MODE, ndim=1)
    test_rdft_accuracy_with_grad(device_target='Ascend', mode=ms.PYNATIVE_MODE, ndim=2)
