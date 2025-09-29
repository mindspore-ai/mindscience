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
# ==============================================================================
''' provide asd fft custom operators python binding '''
import os
import mindspore as ms
from mindspore import nn, ops, dtype as mstype, mint
from mindspore.ops import DataType, CustomRegOp, CustomOpBuilder

_asd_fft_op = None
def _get_asd_fft_op():
    global _asd_fft_op
    if _asd_fft_op is not None:
        return _asd_fft_op
    cur_asd_fft_op_source_file = os.path.join(os.path.dirname(__file__), "asd_fft_op_ext.cpp")
    _asd_fft_op = CustomOpBuilder("asd_fft_op", cur_asd_fft_op_source_file, backend="Ascend", enable_asdsip=True).load()
    return _asd_fft_op

class CustomReal(nn.Cell):
    r"""
    Custom real part extraction operator for complex tensors.

    This operator extracts the real part from complex tensors, used for compatibility
    with different versions of MindSpore.

    Args:
        None

    Inputs:
        - **x** (Tensor): Input complex tensor with data type complex64.

    Outputs:
        - **output** (Tensor): Real part of the input tensor with data type float32.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import CustomReal
        >>> x = ms.Tensor([1+2j, 3+4j], ms.complex64)
        >>> real_op = CustomReal()
        >>> real_part = real_op(x)
        >>> print(real_part)
        [1. 3.]
    """
    def __init__(self):
        super(CustomReal, self).__init__()
        aclnn_ref_info = CustomRegOp("aclnnReal") \
            .input(0, "x", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.C64_Default, DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()

        self.real = ops.Custom(func="aclnnReal", out_shape=lambda x: x,
                               out_dtype=mstype.float32, func_type="aot", reg_info=aclnn_ref_info)
    def construct(self, x):
        return self.real(x)

class CustomComplex(nn.Cell):
    r"""
    Custom complex number construction operator.

    This operator constructs complex tensor from real and imaginary tensor.

    Args:
        auto_prefix (bool): Whether to automatically generate prefix. Default: True.
        flags (dict): Additional flags for the operator. Default: None.

    Inputs:
        - **real** (Tensor): Real part tensor with data type float32.
        - **imag** (Tensor): Imaginary part tensor with data type float32.

    Outputs:
        - **output** (Tensor): Complex tensor with data type complex64.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import CustomComplex
        >>> real = ms.Tensor([1.0, 2.0], ms.float32)
        >>> imag = ms.Tensor([3.0, 4.0], ms.float32)
        >>> complex_op = CustomComplex()
        >>> complex_tensor = complex_op(real, imag)
        >>> print(complex_tensor)
        [1.+3.j 2.+4.j]
    """
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        aclnn_ref_info = CustomRegOp("aclnnComplex") \
            .input(0, "real", "required") \
            .input(1, "imag", "required") \
            .output(0, "out", "required") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.C64_Default) \
            .target("Ascend") \
            .get_op_info()

        self.complex = ops.Custom(func="aclnnComplex", out_shape=lambda real, imag: real,
                                  out_dtype=mstype.complex64, func_type="aot", reg_info=aclnn_ref_info)
    def construct(self, real, imag):
        output = self.complex(real, imag)
        return output


def set_fft_cache_size(cache_size):
    r"""
    Set cache number of ASD FFT operators to optimize the function call performance.
    Without cache, every ASD FFT function call will dlopen the function symbols from .so dynamically,
    which will introduce some overhead.
    With cache, the function symbols will be loaded into cache when the first ASD FFT function call is made,
    and will not be loaded again in subsequent calls.

    Args:
        cache_size (int): Cache number of ASD FFT operators.

    Inputs:
        None

    Outputs:
        None

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> from mindscience.sciops.fft import set_cache_size
        >>> set_cache_size(1024)  # Set 1024 fft operators cache number
    """
    _get_asd_fft_op().asd_set_cache_size(cache_size)


# Use mint.real first, if not exist, fallback to custom operator
try:
    _mint_real = mint.real
    def _get_real(x):
        return _mint_real(x)
except AttributeError:
    _real_op = CustomReal()
    def _get_real(x):
        return _real_op(x)

# Use mint.imag first, if not exist, fallback to custom operator
try:
    _mint_imag = mint.imag
    def _get_imag(x):
        return _mint_imag(x)
except AttributeError:
    _neg1j = ms.tensor(0-1j, dtype=mstype.complex64)
    def _get_imag(x):
        return _get_real(ops.mul(_neg1j, x))

def _get_r2c_alf(xr):
    n = xr.shape[-1]
    m = n // 2 + 1
    k = (n + 1) // 2
    alf = ops.ones(m, xr.dtype)
    # alf[1:k] = 2 but we need scale a/alf, so just set alf[1:k] = 0.5, then do mul(dreal, alf)
    alf[1:k] = 0.5
    return alf

def _get_c2r_alf(xr, scale_factor):
    n = (xr.shape[-1] - 1) * 2
    m = n // 2 + 1
    k = (n + 1) // 2
    alf = ops.ones(m, xr.dtype)
    # for 1->k set alf = 2.0, others set alf = 1.0
    alf[1:k] = 2.0
    alf = alf.mul_(scale_factor)
    return alf



# C2C
class ASD_FFT(nn.Cell): # pylint: disable=invalid-name
    r"""
    1D complex-to-complex forward FFT transform using Ascend NPU acceleration.

    This operator performs 1D Fast Fourier Transform on complex input tensors,
    optimized for Ascend NPU hardware acceleration.

    Args:
        None

    Inputs:
        - **xr** (Tensor): Real part of input complex tensor with data type float32.
        - **xi** (Tensor): Imaginary part of input complex tensor with data type float32.

    Outputs:
        - **yr** (Tensor): Real part of output complex tensor with data type float32.
        - **yi** (Tensor): Imaginary part of output complex tensor with data type float32.

    Raises:
        ValueError: If input tensor data type is not float32.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import ASD_FFT
        >>> xr = ms.Tensor([[1.0, 2.0, 3.0, 4.0]], ms.float32)
        >>> xi = ms.Tensor([[0.0, 0.0, 0.0, 0.0]], ms.float32)
        >>> asd_fft = ASD_FFT()
        >>> yr, yi = asd_fft(xr, xi)
        >>> print(yr.shape)
        (1, 4)
        >>> print(yi.shape)
        (1, 4)
    """
    def __init__(self):
        super(ASD_FFT, self).__init__()
        self.asd_fft_op = _get_asd_fft_op().asd_fft_1d
        self.make_complex = CustomComplex()
        self.used_bprop_inputs = []

    def get_fft_size_and_scale(self, xr):
        return xr.shape[-1], None


    def construct(self, xr, xi):
        return self.forward(xr, xi)

    def forward(self, xr, xi=None):
        """forward"""
        if xr.dtype != mstype.float32 or (xi is not None and xi.dtype != mstype.float32):
            raise ValueError("ASD_FFT Input tensor must be float32")
        org_shape = list(xr.shape)

        # Unify to two dimensions
        batch_size = 1
        for i in range(len(org_shape) - 1):
            batch_size *= org_shape[i]
        if len(org_shape) != 2:
            xr = mint.reshape(xr, (batch_size, org_shape[-1]))
            xi = mint.reshape(xi, (batch_size, org_shape[-1])) if xi is not None else None

        fft_size, scale_factor = self.get_fft_size_and_scale(xr)
        x = self.make_complex(xr, xi) if xi is not None else xr
        output = self.asd_fft_op(x, xr.shape[0], fft_size)

        if scale_factor is not None:
            output.mul_(scale_factor)

        if org_shape != list(output.shape):
            org_shape[-1] = output.shape[-1]
            output = mint.reshape(output, tuple(org_shape))
        # If output is not complex, return directly
        if not ops.is_complex(output):
            return output

        return _get_real(output), _get_imag(output)

    def bprop(self, xr, xi, out, dout): # pylint: disable=unused-argument
        dreal, dimag = dout
        dxr, dxi = asd_ifft(dreal, dimag)
        return dxr.mul_(dreal.shape[-1]), dxi.mul_(dreal.shape[-1])

# C2C
class ASD_IFFT(ASD_FFT): # pylint: disable=invalid-name
    r"""
    1D complex-to-complex inverse FFT transform using Ascend NPU acceleration.

    This operator performs 1D Inverse Fast Fourier Transform on complex input tensors,
    optimized for Ascend NPU hardware acceleration.

    Args:
        None

    Inputs:
        - **xr** (Tensor): Real part of input complex tensor with data type float32.
        - **xi** (Tensor): Imaginary part of input complex tensor with data type float32.

    Outputs:
        - **yr** (Tensor): Real part of output complex tensor with data type float32.
        - **yi** (Tensor): Imaginary part of output complex tensor with data type float32.

    Raises:
        ValueError: If input tensor data type is not float32.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import ASD_IFFT
        >>> xr = ms.Tensor([[1.0, 2.0, 3.0, 4.0]], ms.float32)
        >>> xi = ms.Tensor([[0.0, 0.0, 0.0, 0.0]], ms.float32)
        >>> asd_ifft = ASD_IFFT()
        >>> yr, yi = asd_ifft(xr, xi)
        >>> print(yr.shape)
        (1, 4)
        >>> print(yi.shape)
        (1, 4)
    """
    def __init__(self):
        super(ASD_IFFT, self).__init__()
        self.asd_fft_op = _get_asd_fft_op().asd_ifft_1d
        self.used_bprop_inputs = []

    def get_fft_size_and_scale(self, xr):
        fft_size = xr.shape[-1]
        scale_factor = 1.0 / fft_size
        return fft_size, scale_factor

    def bprop(self, xr, xi, out, dout): # pylint: disable=unused-argument
        dreal, dimag = dout
        dxr, dxi = asd_fft(dreal, dimag)
        scale_factor = 1.0 / dreal.shape[-1]
        return dxr.mul_(scale_factor), dxi.mul_(scale_factor)

# R2C
class ASD_RFFT(ASD_FFT): # pylint: disable=invalid-name
    r"""
    1D real-to-complex FFT transform using Ascend NPU acceleration.

    This operator performs 1D Real Fast Fourier Transform on real input tensors,
    optimized for Ascend NPU hardware acceleration.

    Args:
        None

    Inputs:
        - **xr** (Tensor): Input real tensor with data type float32.

    Outputs:
        - **yr** (Tensor): Real part of output complex tensor with data type float32.
        - **yi** (Tensor): Imaginary part of output complex tensor with data type float32.

    Raises:
        ValueError: If input tensor data type is not float32.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import ASD_RFFT
        >>> xr = ms.Tensor([[1.0, 2.0, 3.0, 4.0]], ms.float32)
        >>> asd_rfft = ASD_RFFT()
        >>> yr, yi = asd_rfft(xr)
        >>> print(yr.shape)
        (1, 3)
        >>> print(yi.shape)
        (1, 3)
    """
    def __init__(self):
        super(ASD_RFFT, self).__init__()
        self.asd_fft_op = _get_asd_fft_op().asd_rfft_1d
        self.used_bprop_inputs = [0]

    def construct(self, xr): # pylint: disable=arguments-differ
        return self.forward(xr)

    def bprop(self, xr, out, dout): # pylint: disable=arguments-differ, unused-argument
        dreal, dimag = dout
        alf = _get_r2c_alf(xr)
        dxr = asd_irfftn(dreal.mul_(alf), dimag.mul_(alf), ndim=1, n=xr.shape[-1])
        return dxr.mul_(xr.shape[-1])

# C2R
class ASD_IRFFT(ASD_FFT): # pylint: disable=invalid-name
    r"""
    1D complex-to-real inverse FFT transform using Ascend NPU acceleration.

    This operator performs 1D Inverse Real Fast Fourier Transform on complex input tensors,
    optimized for Ascend NPU hardware acceleration.

    Args:
        None

    Inputs:
        - **xr** (Tensor): Real part of input complex tensor with data type float32.
        - **xi** (Tensor): Imaginary part of input complex tensor with data type float32.

    Outputs:
        - **yr** (Tensor): Output real tensor with data type float32.

    Raises:
        ValueError: If input tensor data type is not float32.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import ASD_IRFFT
        >>> xr = ms.Tensor([[1.0, 2.0, 3.0]], ms.float32)
        >>> xi = ms.Tensor([[0.0, 0.0, 0.0]], ms.float32)
        >>> asd_irfft = ASD_IRFFT()
        >>> yr = asd_irfft(xr, xi)
        >>> print(yr.shape)
        (1, 4)
    """
    def __init__(self):
        super(ASD_IRFFT, self).__init__()
        self.asd_fft_op = _get_asd_fft_op().asd_irfft_1d
        self.n = None
        self.used_bprop_inputs = [0]

    def get_fft_size_and_scale(self, xr):
        if self.n is not None and (self.n // 2 + 1) == xr.shape[-1]:
            fft_size = self.n
        else:
            fft_size = (xr.shape[-1] - 1) * 2
        scale_factor = 1.0 / fft_size
        return fft_size, scale_factor

    def set_n(self, n):
        self.n = n

    def bprop(self, xr, xi, out, dout): # pylint: disable=unused-argument
        dreal = dout
        fft_size, scale_factor = self.get_fft_size_and_scale(xr) # pylint: disable=unused-variable
        alf = _get_c2r_alf(xr, scale_factor)
        dxr, dxi = asd_rfft(dreal)

        return dxr.mul_(alf), dxi.mul_(alf)

# C2C forward
class ASD_FFT2D(nn.Cell): # pylint: disable=invalid-name
    r"""
    2D complex-to-complex forward FFT transform using Ascend NPU acceleration.

    This operator performs 2D Fast Fourier Transform on complex input tensors,
    optimized for Ascend NPU hardware acceleration.

    Args:
        None

    Inputs:
        - **xr** (Tensor): Real part of input complex tensor with data type float32, at least 2D.
        - **xi** (Tensor): Imaginary part of input complex tensor with data type float32.

    Outputs:
        - **yr** (Tensor): Real part of output complex tensor with data type float32.
        - **yi** (Tensor): Imaginary part of output complex tensor with data type float32.

    Raises:
        ValueError: If input tensor data type is not float32 or tensor has less than 2 dimensions.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import ASD_FFT2D
        >>> xr = ms.Tensor([[[1.0, 2.0], [3.0, 4.0]]], ms.float32)
        >>> xi = ms.Tensor([[[0.0, 0.0], [0.0, 0.0]]], ms.float32)
        >>> asd_fft2d = ASD_FFT2D()
        >>> yr, yi = asd_fft2d(xr, xi)
        >>> print(yr.shape)
        (1, 2, 2)
        >>> print(yi.shape)
        (1, 2, 2)
    """
    def __init__(self):
        super(ASD_FFT2D, self).__init__()
        self.asd_fft_op = _get_asd_fft_op().asd_fft_2d
        self.make_complex = CustomComplex()
        self.used_bprop_inputs = []

    def get_fft_size_and_scale(self, xr):
        return xr.shape[0], xr.shape[1], xr.shape[2], None

    def construct(self, xr, xi):
        return self.forward(xr, xi)

    def forward(self, xr, xi=None):
        """forward"""
        if xr.dtype != mstype.float32 or (xi is not None and xi.dtype != mstype.float32):
            raise ValueError("ASD_FFT Input tensor must be float32")

        org_shape = list(xr.shape)
        if len(org_shape) < 2:
            raise ValueError("2D FFT Input tensor must have at least 2 dimensions")
        # Unify to three dimensions
        if len(org_shape) == 2:
            xr = ops.expand_dims(xr, 0)
            xi = ops.expand_dims(xi, 0) if xi is not None else None
        elif len(org_shape) > 3:
            xr = ops.reshape(xr, (-1, org_shape[-2], org_shape[-1]))
            xi = ops.reshape(xi, (-1, org_shape[-2], org_shape[-1])) if xi is not None else None

        batch_size, x_size, y_size, scale_factor = self.get_fft_size_and_scale(xr)
        x = self.make_complex(xr, xi) if xi is not None else xr
        output = self.asd_fft_op(x, batch_size, x_size, y_size)

        if scale_factor is not None:
            output.mul_(scale_factor)

        if org_shape != list(output.shape):
            org_shape[-1] = output.shape[-1]
            output = ops.reshape(output, tuple(org_shape))

        # If output is not complex, return directly
        if not ops.is_complex(output):
            return output

        return _get_real(output), _get_imag(output)

    def bprop(self, xr, xi, out, dout): # pylint: disable=unused-argument
        dreal, dimag = dout
        dxr, dxi = asd_ifft2(dreal, dimag)
        n = dreal.shape[-1] * dreal.shape[-2]
        return dxr.mul_(n), dxi.mul_(n)

# C2C inverse
class ASD_IFFT2D(ASD_FFT2D): # pylint: disable=invalid-name
    r"""
    2D complex-to-complex inverse FFT transform using Ascend NPU acceleration.

    This operator performs 2D Inverse Fast Fourier Transform on complex input tensors,
    optimized for Ascend NPU hardware acceleration.

    Args:
        None

    Inputs:
        - **xr** (Tensor): Real part of input complex tensor with data type float32, at least 2D.
        - **xi** (Tensor): Imaginary part of input complex tensor with data type float32.

    Outputs:
        - **yr** (Tensor): Real part of output complex tensor with data type float32.
        - **yi** (Tensor): Imaginary part of output complex tensor with data type float32.

    Raises:
        ValueError: If input tensor data type is not float32 or tensor has less than 2 dimensions.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import ASD_IFFT2D
        >>> xr = ms.Tensor([[[1.0, 2.0], [3.0, 4.0]]], ms.float32)
        >>> xi = ms.Tensor([[[0.0, 0.0], [0.0, 0.0]]], ms.float32)
        >>> asd_ifft2d = ASD_IFFT2D()
        >>> yr, yi = asd_ifft2d(xr, xi)
        >>> print(yr.shape)
        (1, 2, 2)
        >>> print(yi.shape)
        (1, 2, 2)
    """
    def __init__(self):
        super(ASD_IFFT2D, self).__init__()
        self.asd_fft_op = _get_asd_fft_op().asd_ifft_2d
        self.used_bprop_inputs = []

    def get_fft_size_and_scale(self, xr):
        batch_size, x_size, y_size, scale_factor = super(ASD_IFFT2D, self).get_fft_size_and_scale(xr)
        scale_factor = 1.0 / (x_size * y_size)
        return batch_size, x_size, y_size, scale_factor

    def bprop(self, xr, xi, out, dout): # pylint: disable=unused-argument
        dreal, dimag = dout
        dxr, dxi = asd_fft2(dreal, dimag)
        scale_factor = 1.0 / (dreal.shape[-1] * dreal.shape[-2])
        return dxr.mul_(scale_factor), dxi.mul_(scale_factor)

# R2C forward
class ASD_RFFT2D(ASD_FFT2D): # pylint: disable=invalid-name
    r"""
    2D real-to-complex FFT transform using Ascend NPU acceleration.

    This operator performs 2D Real Fast Fourier Transform on real input tensors,
    optimized for Ascend NPU hardware acceleration.

    Args:
        None

    Inputs:
        - **xr** (Tensor): Input real tensor with data type float32, at least 2D.

    Outputs:
        - **yr** (Tensor): Real part of output complex tensor with data type float32.
        - **yi** (Tensor): Imaginary part of output complex tensor with data type float32.

    Raises:
        ValueError: If input tensor data type is not float32 or tensor has less than 2 dimensions.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import ASD_RFFT2D
        >>> xr = ms.Tensor([[[1.0, 2.0], [3.0, 4.0]]], ms.float32)
        >>> asd_rfft2d = ASD_RFFT2D()
        >>> yr, yi = asd_rfft2d(xr)
        >>> print(yr.shape)
        (1, 2, 2)
        >>> print(yi.shape)
        (1, 2, 2)
    """
    def __init__(self):
        super(ASD_RFFT2D, self).__init__()
        self.asd_fft_op = _get_asd_fft_op().asd_rfft_2d
        self.used_bprop_inputs = [0]

    def construct(self, xr): # pylint: disable=arguments-differ
        return self.forward(xr)

    def bprop(self, xr, out, dout): # pylint: disable=arguments-differ, unused-argument
        dreal, dimag = dout
        alf = _get_r2c_alf(xr)
        dreal.mul_(alf)
        dimag.mul_(alf)
        dxr = asd_irfftn(dreal, dimag, ndim=2, n=xr.shape[-1])
        n = xr.shape[-1] * xr.shape[-2] * 1.0
        return dxr.mul_(n)

# C2R inverse
class ASD_IRFFT2D(ASD_FFT2D): # pylint: disable=invalid-name
    r"""
    2D complex-to-real inverse FFT transform using Ascend NPU acceleration.

    This operator performs 2D Inverse Real Fast Fourier Transform on complex input tensors,
    optimized for Ascend NPU hardware acceleration.

    Args:
        None

    Inputs:
        - **xr** (Tensor): Real part of input complex tensor with data type float32, at least 2D.
        - **xi** (Tensor): Imaginary part of input complex tensor with data type float32.

    Outputs:
        - **yr** (Tensor): Output real tensor with data type float32.

    Raises:
        ValueError: If input tensor data type is not float32 or tensor has less than 2 dimensions.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import ASD_IRFFT2D
        >>> xr = ms.Tensor([[[1.0, 2.0], [3.0, 4.0]]], ms.float32)
        >>> xi = ms.Tensor([[[0.0, 0.0], [0.0, 0.0]]], ms.float32)
        >>> asd_irfft2d = ASD_IRFFT2D()
        >>> yr = asd_irfft2d(xr, xi)
        >>> print(yr.shape)
        (1, 2, 4)
    """
    def __init__(self):
        super(ASD_IRFFT2D, self).__init__()
        self.asd_fft_op = _get_asd_fft_op().asd_irfft_2d
        self.n = None
        self.used_bprop_inputs = [0]

    def get_fft_size_and_scale(self, xr):
        batch_size = xr.shape[0]
        x_size = xr.shape[1]
        y_size = xr.shape[2]
        if self.n is not None and (self.n // 2 + 1) == y_size:
            output_last_dim = self.n
        else:
            output_last_dim = (y_size - 1) * 2
        scale_factor = 1.0 / (x_size * output_last_dim)
        return batch_size, x_size, output_last_dim, scale_factor

    def set_n(self, n):
        self.n = n

    def bprop(self, xr, xi, out, dout): # pylint: disable=unused-argument
        dreal = dout
        dxr, dxi = asd_rfft2(dreal)
        x_size = xr.shape[-2]
        y_size = xr.shape[-1]
        if self.n is not None and (self.n // 2 + 1) == y_size:
            output_last_dim = self.n
        else:
            output_last_dim = (y_size - 1) * 2
        scale_factor = 1.0 / (x_size * output_last_dim)
        alf = _get_c2r_alf(xr, scale_factor)
        return dxr.mul_(alf), dxi.mul_(alf)

# 全局变量用于存储单例实例
_asd_fft_instance = None
_asd_ifft_instance = None
_asd_rfft_instance = None
_asd_irfft_instance = None
_asd_fft2_instance = None
_asd_ifft2_instance = None
_asd_rfft2_instance = None
_asd_irfft2_instance = None

# 延迟初始化的FFT操作符函数
def asd_fft(*args, **kwargs):
    """延迟初始化的ASD_FFT调用"""
    global _asd_fft_instance
    if _asd_fft_instance is None:
        _asd_fft_instance = ASD_FFT()
    return _asd_fft_instance(*args, **kwargs)

def asd_ifft(*args, **kwargs):
    """延迟初始化的ASD_IFFT调用"""
    global _asd_ifft_instance
    if _asd_ifft_instance is None:
        _asd_ifft_instance = ASD_IFFT()
    return _asd_ifft_instance(*args, **kwargs)

def asd_rfft(*args, **kwargs):
    """延迟初始化的ASD_RFFT调用"""
    global _asd_rfft_instance
    if _asd_rfft_instance is None:
        _asd_rfft_instance = ASD_RFFT()
    return _asd_rfft_instance(*args, **kwargs)

def asd_irfft(*args, **kwargs):
    """延迟初始化的ASD_IRFFT调用"""
    global _asd_irfft_instance
    if _asd_irfft_instance is None:
        _asd_irfft_instance = ASD_IRFFT()
    return _asd_irfft_instance(*args, **kwargs)

def asd_fft2(*args, **kwargs):
    """延迟初始化的ASD_FFT2D调用"""
    global _asd_fft2_instance
    if _asd_fft2_instance is None:
        _asd_fft2_instance = ASD_FFT2D()
    return _asd_fft2_instance(*args, **kwargs)

def asd_ifft2(*args, **kwargs):
    """延迟初始化的ASD_IFFT2D调用"""
    global _asd_ifft2_instance
    if _asd_ifft2_instance is None:
        _asd_ifft2_instance = ASD_IFFT2D()
    return _asd_ifft2_instance(*args, **kwargs)

def asd_rfft2(*args, **kwargs):
    """延迟初始化的ASD_RFFT2D调用"""
    global _asd_rfft2_instance
    if _asd_rfft2_instance is None:
        _asd_rfft2_instance = ASD_RFFT2D()
    return _asd_rfft2_instance(*args, **kwargs)

def asd_irfft2(*args, **kwargs):
    """延迟初始化的ASD_IRFFT2D调用"""
    global _asd_irfft2_instance
    if _asd_irfft2_instance is None:
        _asd_irfft2_instance = ASD_IRFFT2D()
    return _asd_irfft2_instance(*args, **kwargs)

def asd_fftn(xr, xi, ndim=1):
    r"""
    N-dimensional complex-to-complex forward FFT transform using Ascend NPU acceleration.

    This function provides a unified interface for 1D and 2D complex-to-complex FFT transforms,
    optimized for Ascend NPU hardware acceleration.

    Args:
        xr (Tensor): Real part of input complex tensor with data type float32.
        xi (Tensor): Imaginary part of input complex tensor with data type float32.
        ndim (int): Number of dimensions to transform. Default: 1. Only support 1 and 2.

    Inputs:
        - **xr** (Tensor): Real part of input complex tensor with data type float32.
        - **xi** (Tensor): Imaginary part of input complex tensor with data type float32.
        - **ndim** (int): Number of dimensions to transform. Default: 1.

    Outputs:
        - **yr** (Tensor): Real part of output complex tensor with data type float32.
        - **yi** (Tensor): Imaginary part of output complex tensor with data type float32.

    Raises:
        ValueError: If ndim is not 1 or 2.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import asd_fftn
        >>> xr = ms.Tensor([[1.0, 2.0, 3.0, 4.0]], ms.float32)
        >>> xi = ms.Tensor([[0.0, 0.0, 0.0, 0.0]], ms.float32)
        >>> yr, yi = asd_fftn(xr, xi, ndim=1)
        >>> print(yr.shape)
        (1, 4)
        >>> print(yi.shape)
        (1, 4)
    """
    if ndim == 1:
        return asd_fft(xr, xi)
    if ndim == 2:
        return asd_fft2(xr, xi)
    raise ValueError(f"asd_fftn Unsupported dimension: {ndim}, only support 1D and 2D")

def asd_ifftn(xr, xi, ndim=1):
    r"""
    N-dimensional complex-to-complex inverse FFT transform using Ascend NPU acceleration.

    This function provides a unified interface for 1D and 2D complex-to-complex inverse FFT transforms,
    optimized for Ascend NPU hardware acceleration.

    Args:
        xr (Tensor): Real part of input complex tensor with data type float32.
        xi (Tensor): Imaginary part of input complex tensor with data type float32.
        ndim (int): Number of dimensions to transform. Default: 1. Only support 1 and 2.

    Inputs:
        - **xr** (Tensor): Real part of input complex tensor with data type float32.
        - **xi** (Tensor): Imaginary part of input complex tensor with data type float32.
        - **ndim** (int): Number of dimensions to transform. Default: 1.

    Outputs:
        - **yr** (Tensor): Real part of output complex tensor with data type float32.
        - **yi** (Tensor): Imaginary part of output complex tensor with data type float32.

    Raises:
        ValueError: If ndim is not 1 or 2.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import asd_ifftn
        >>> xr = ms.Tensor([[1.0, 2.0, 3.0, 4.0]], ms.float32)
        >>> xi = ms.Tensor([[0.0, 0.0, 0.0, 0.0]], ms.float32)
        >>> yr, yi = asd_ifftn(xr, xi, ndim=1)
        >>> print(yr.shape)
        (1, 4)
        >>> print(yi.shape)
        (1, 4)
    """
    if ndim == 1:
        return asd_ifft(xr, xi)
    if ndim == 2:
        return asd_ifft2(xr, xi)
    raise ValueError(f"asd_ifftn Unsupported dimension: {ndim}, only support 1D and 2D")

def asd_rfftn(xr, ndim=1):
    r"""
    N-dimensional real-to-complex FFT transform using Ascend NPU acceleration.

    This function provides a unified interface for 1D and 2D real-to-complex FFT transforms,
    optimized for Ascend NPU hardware acceleration.

    Args:
        xr (Tensor): Input real tensor with data type float32.
        ndim (int): Number of dimensions to transform. Default: 1. Only support 1 and 2.

    Inputs:
        - **xr** (Tensor): Input real tensor with data type float32.
        - **ndim** (int): Number of dimensions to transform. Default: 1.

    Outputs:
        - **yr** (Tensor): Real part of output complex tensor with data type float32.
        - **yi** (Tensor): Imaginary part of output complex tensor with data type float32.

    Raises:
        ValueError: If ndim is not 1 or 2.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import asd_rfftn
        >>> xr = ms.Tensor([[1.0, 2.0, 3.0, 4.0]], ms.float32)
        >>> yr, yi = asd_rfftn(xr, ndim=1)
        >>> print(yr.shape)
        (1, 3)
        >>> print(yi.shape)
        (1, 3)
    """
    if ndim == 1:
        return asd_rfft(xr)
    if ndim == 2:
        return asd_rfft2(xr)
    raise ValueError(f"asd_rfftn Unsupported dimension: {ndim}, only support 1D and 2D")

def asd_irfftn(xr, xi, n=None, ndim=1):
    r"""
    N-dimensional complex-to-real inverse FFT transform using Ascend NPU acceleration.

    This function provides a unified interface for 1D and 2D complex-to-real inverse FFT transforms,
    optimized for Ascend NPU hardware acceleration.

    Args:
        xr (Tensor): Real part of input complex tensor with data type float32.
        xi (Tensor): Imaginary part of input complex tensor with data type float32.
        n (int, optional): Length of the output tensor. Default: None.
        ndim (int): Number of dimensions to transform. Default: 1. Only support 1 and 2.

    Inputs:
        - **xr** (Tensor): Real part of input complex tensor with data type float32.
        - **xi** (Tensor): Imaginary part of input complex tensor with data type float32.
        - **n** (int, optional): Length of the output tensor. Default: None.
        - **ndim** (int): Number of dimensions to transform. Default: 1.

    Outputs:
        - **yr** (Tensor): Output real tensor with data type float32.

    Raises:
        ValueError: If ndim is not 1 or 2.

    Supported Platforms:
        ``Ascend`` ``Pynative``

    Examples:
        >>> import mindspore as ms
        >>> from mindscience.sciops.fft import asd_irfftn
        >>> xr = ms.Tensor([[1.0, 2.0, 3.0]], ms.float32)
        >>> xi = ms.Tensor([[0.0, 0.0, 0.0]], ms.float32)
        >>> yr = asd_irfftn(xr, xi, ndim=1)
        >>> print(yr.shape)
        (1, 4)
    """
    if ndim == 1:
        instance = ASD_IRFFT()
        instance.set_n(n)
        return instance(xr, xi)
    if ndim == 2:
        instance = ASD_IRFFT2D()
        instance.set_n(n)
        return instance(xr, xi)
    raise ValueError(f"asd_irfftn Unsupported dimension: {ndim}, only support 1D and 2D")
