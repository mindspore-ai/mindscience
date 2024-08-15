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

from mindspore.nn import Cell
from mindspore import ops

__all__ = ['FFT3DAD',
           'IFFT3DAD',
           'fft3d',
           'ifft3d',
           ]

# pylint: disable=invalid-name


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

try:
    fft3d = ops.FFTWithSize(signal_ndim=3, real=True, inverse=False)
    ifft3d = ops.FFTWithSize(signal_ndim=3, real=True, inverse=True, norm="forward")
    mw_is_vaild = True
except AttributeError:
    fft3d = FFT3DAD()
    ifft3d = IFFT3DAD()
    mw_is_vaild = False
