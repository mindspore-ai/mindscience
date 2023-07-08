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
from mindspore.ops import DataType, CustomRegOp


class FFTOP():
    """Class for registering FFT Operators"""

    def __init__(self):
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../libs/libfft.so"))
        self.fft3d_func = lib_path + ":FFT3D"
        self.ifft3d_func = lib_path + ":IFFT3D"
        self.fft3d_reg_info = CustomRegOp() \
            .input(0, "input_tensor") \
            .output(0, "output_tensor") \
            .dtype_format(DataType.F32_Default, DataType.C64_Default) \
            .target("GPU") \
            .get_op_info()
        self.ifft3d_reg_info = CustomRegOp() \
            .input(0, "input_tensor") \
            .output(0, "output_tensor") \
            .dtype_format(DataType.C64_Default, DataType.F32_Default) \
            .target("GPU") \
            .get_op_info()

    @staticmethod
    def fft3d_infer_shape(input_shape):
        return [input_shape[0], input_shape[1], int(input_shape[2] / 2) + 1]

    @staticmethod
    def fft3d_infer_dtype(input_dtype):
        # pylint: disable=unused-argument
        return mstype.complex64

    @staticmethod
    def ifft3d_infer_shape(input_shape):
        return [input_shape[0], input_shape[1], (input_shape[2] - 1) * 2]

    @staticmethod
    def ifft3d_infer_dtype(input_dtype):
        # pylint: disable=unused-argument
        return mstype.float32

    def register(self):
        fft3d_op = ops.Custom(self.fft3d_func, out_shape=self.fft3d_infer_shape, out_dtype=self.fft3d_infer_dtype,
                              func_type="aot", reg_info=self.fft3d_reg_info)
        ifft3d_op = ops.Custom(self.ifft3d_func, out_shape=self.ifft3d_infer_shape, out_dtype=self.ifft3d_infer_dtype,
                               func_type="aot", reg_info=self.ifft3d_reg_info)
        return fft3d_op, ifft3d_op
