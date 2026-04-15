# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Custom functions2
"""

import os
import mindspore as ms
from mindspore.nn import Cell
from mindspore import ops
from mindspore import log as logger

_cur_file_dir, _ = os.path.split(os.path.realpath(__file__))
_fft_file = _cur_file_dir + '/../ccsrc/FFT.so'

__all__ = ['customRFFT3D',
           'customIRFFT3D',
           'FFT3DAD',
           'IFFT3DAD',
           'fft3d',
           'ifft3d',
           ]

# pylint: disable=invalid-name


def customRFFT3D():
    """
    Forward FFT with Three-Dimensional Input.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **input_tensor** (Tensor) - Three dimensional tensor, supported
          data type is float32.

    Outputs:
        - **output_tensor** (Tensor) - The tensor after undergoing fast Fourier
          transform, the data type is complex64.

    Supported Platforms:
        ``GPU``
    """

    def bprop():
        op = ops.Custom(_fft_file + ":CustomIRFFT3D", out_shape=lambda x: (x[0], x[1], (x[2]-1)*2),
                        out_dtype=ms.float32, func_type="aot")

        def custom_bprop(x, out, dout):
        # pylint: disable=unused-argument
            dx = op(dout)
            return (dx,)
        return custom_bprop
    op = ops.Custom(_fft_file + ":CustomRFFT3D", out_shape=lambda x: (x[0], x[1], x[2]//2+1),
                    out_dtype=ms.complex64, bprop=bprop(), func_type="aot")
    return op


def customIRFFT3D():
    """
    Inverse FFT with Three-Dimensional Input.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **input_tensor** (Tensor) - Three dimensional input tensor, supported data
          type is complex64.

    Outputs:
        - **output_tensor** (Tensor) - Returns the tensor after undergoing
          inverse Fourier transform, the data type is float32.

    Supported Platforms:
        ``GPU``
    """
    def bprop():
        op = ops.Custom(_fft_file + ":CustomRFFT3D", out_shape=lambda x: (
            x[0], x[1], x[2]//2+1), out_dtype=ms.complex64, func_type="aot")

        def custom_bprop(x, out, dout):
        # pylint: disable=unused-argument
            dx = op(dout)
            return (dx,)
        return custom_bprop
    op = ops.Custom(_fft_file + ":CustomIRFFT3D", out_shape=lambda x: (x[0], x[1], (x[2]-1)*2),
                    out_dtype=ms.float32, bprop=bprop(), func_type="aot")
    return op


class FFT3DAD(Cell):
    r"""FFT operator with automatically differenatiable"""

    def __init__(self):
        Cell.__init__(self)

        self.rfft3d = ops.FFT3D()
        self.irfft3d = ops.IFFT3D()

    def construct(self, x):
        return self.rfft3d(x)

    def bprop(self, x, out, dout):
        #pylint: disable=unused-argument
        return (self.irfft3d(dout),)


class IFFT3DAD(Cell):
    r"""IFFT operator with automatically differenatiable"""

    def __init__(self):
        Cell.__init__(self)

        self.rfft3d = ops.FFT3D()
        self.irfft3d = ops.IFFT3D()

    def construct(self, x):
        return self.irfft3d(x)

    def bprop(self, x, out, dout):
        # pylint: disable=unused-argument
        return (self.rfft3d(dout),)


if os.path.exists(_fft_file):
    fft3d = customRFFT3D()
    ifft3d = customIRFFT3D()
    mw_is_vaild = False
else:
    logger.info("Cannot find the FFT.so file, will use MindSpore's FFT operator, which may cause bugs.")
    try:
        fft3d = ops.FFTWithSize(signal_ndim=3, real=True, inverse=False)
        ifft3d = ops.FFTWithSize(signal_ndim=3, real=True, inverse=True, norm="forward")
        mw_is_vaild = True
    except AttributeError:
        fft3d = FFT3DAD()
        ifft3d = IFFT3DAD()
        mw_is_vaild = False
