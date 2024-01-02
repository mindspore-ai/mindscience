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
"""FFT Operators"""

import os
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore import nn


_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../libs/libfft.so"))

class RFFT3D(nn.Cell):
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

    def __init__(self):
        super().__init__()
        def bprop():
            op = ops.Custom(_lib_path + ":IFFT3D", out_shape=lambda x: (x[0], x[1], (x[2]-1)*2),
                            out_dtype=mstype.float32, func_type="aot")
            def custom_bprop(x, out, dout):
            # pylint: disable=unused-argument
                dx = op(dout)
                return (dx,)
            return custom_bprop
        self.op = ops.Custom(_lib_path + ":FFT3D", out_shape=lambda x: (x[0], x[1], x[2]//2+1),
                             out_dtype=mstype.complex64, bprop=bprop(), func_type="aot")

    def construct(self, x):
        return self.op(x)

class IRFFT3D(nn.Cell):
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

    def __init__(self):
        super().__init__()
        def bprop():
            op = ops.Custom(_lib_path + ":FFT3D", out_shape=lambda x: (x[0], x[1], x[2]//2+1),
                            out_dtype=mstype.complex64, func_type="aot")
            def custom_bprop(x, out, dout):
            # pylint: disable=unused-argument
                dx = op(dout)
                return (dx,)
            return custom_bprop
        self.op = ops.Custom(_lib_path + ":IFFT3D", out_shape=lambda x: (x[0], x[1], (x[2]-1)*2),
                             out_dtype=mstype.float32, bprop=bprop(), func_type="aot")

    def construct(self, x):
        return self.op(x)
