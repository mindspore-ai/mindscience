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

"""Common modules."""

import numpy as np
import mindspore as ms
from mindspore import nn, mint
from mindspore.common import initializer
from mindscience.e3nn.utils import Ncon
from mindscience.sciops.einsum import Einsum

# Useful for mocking in tests.
DEFAULT_PRECISION = None

# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(
    0.87962566103423978, dtype=np.float32
)


class LayerNorm(nn.Cell):
    """LayerNorm module.

    Equivalent to ms.nn.LayerNorm. In most cases, it can be replaced by ms.nn.LayerNorm. 
    Here, gamma is scale, beta is shift or offset
    Args:
        normalized_shape (tuple | list): The shape of Tensor which need to LayerNorm.
        name (str): Name of this layer.
        begin_norm_axis(int): From which axis norm begin
        begin_params_axis(int): From which axis params begin
        gamma_init('str'): Initializer of gamma
        beta_init('str'): Initializer of beta
        epsilon(float): epsilon value
        dtype(ms.type): Type of output
        create_beta(bool): whether to create a trainable beta parameter
        create_gamma(bool): whether to create a trainable gamma parameter
    Inputs:
        - **x** (Tensor) - Tensor of any shape
    Outputs:
        The shape of tensor is the same as x.
    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, normalized_shape, name=None, begin_norm_axis=-1,  # pylint: disable=unused-argument
                 begin_params_axis=-1, gamma_init='ones',  # pylint: disable=unused-argument
                 beta_init='zeros', epsilon=1e-5, dtype=ms.float32,  # pylint: disable=unused-argument
                 create_beta=True, create_gamma=True):
        super().__init__()
        if not create_beta:
            beta_init = 'zeros'
        if not create_gamma:
            gamma_init = 'ones'
        self.layernorm = LayerNormLayer(
            normalized_shape[begin_norm_axis:], create_gamma, create_beta, epsilon, gamma_init, beta_init)

    def construct(self, x):
        out = self.layernorm(x.astype(ms.float32)).astype(x.dtype)
        return out


class LayerNormLayer(nn.Cell):
    """"LayernormLayer"""
    def __init__(self, norm_shape, create_gamma, create_beta, eps=1e-5, gamma_init='ones', beta_init='zeros'):
        super().__init__()
        self.norm_shape = norm_shape
        self.eps = eps
        if create_gamma:
            self.weight = ms.Parameter(initializer.initializer(
                gamma_init, self.norm_shape, dtype=ms.float32))
        else:
            self.weight = ms.Tensor(ms.Parameter(initializer.initializer(
                gamma_init, self.norm_shape, dtype=ms.float32)))
        if create_beta:
            self.bias = ms.Parameter(initializer.initializer(
                beta_init, self.norm_shape, dtype=ms.float32))
        else:
            self.bias = ms.Tensor(ms.Parameter(initializer.initializer(
                beta_init, self.norm_shape, dtype=ms.float32)))

    def construct(self, x):
        """LayerNormLayer"""
        out = ms.ops.layer_norm(
            x, self.norm_shape, self.weight, self.bias, self.eps)
        return out


class LinearAMP(nn.Cell):
    """
    LinearAMP: Linear with Automatic Mixed Precision.
    Args:
        in_channels (int): input channels.
        out_channels (int): output channels.
        weight_init (str): weight initializer, default is None.
        bias_init (str): bias initializer, default is None.
        has_bias (bool): whether to have bias, default is True.
        activation (str): activation function, default is None.
        dtype (ms.dtype): dtype, default is ms.float32.
        name (str): name, default is None.
    Inputs:
        - **x** (Tensor)
    Outputs:
        The shape of tensor is the same as x.
    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, in_channels, out_channels, weight_init=None,
                 bias_init=None, has_bias=True, activation=None,
                 dtype=ms.float32, name=None):
        super().__init__(self)
        self.dtype = dtype
        if name is not None:
            self.name = name
        else:
            self.name = None
        self.linear = nn.Dense(in_channels, out_channels, weight_init=weight_init,
                               bias_init=bias_init, has_bias=has_bias, activation=activation, dtype=ms.float32)

    def construct(self, x):
        """LinearAMP"""
        if self.dtype == ms.float32:
            out = self.linear(x)
        else:
            x = x.astype(self.dtype)
            if self.linear.bias is not None:
                out = mint.matmul(x, self.linear.weight.T.astype(
                    self.dtype)) + self.linear.bias.astype(self.dtype)
            else:
                out = mint.matmul(x, self.linear.weight.T.astype(self.dtype))
        return out


class CustomDense(nn.Cell):
    """
    Custom Linear Module. It can be apply to a high dimension Tensor, and can be used on more than 1D Matmul.
    In Alphafold, they use Einsum to replace Matmul, here we use Ncon to replace Matmul. if in_shape and out_shape
    are both int, this layer is equivalence to nn.Dense.
    Args:
        in_shape (Union(int, List, Tuple)): input shape, that need to be multiplied.
        out_shape (Union(int, List, Tuple)): output shape, that need to be multiplied.
    Inputs:
        - **x** (Tensor)
    Outputs:

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, in_shape, out_shape, weight_init="ones",
                 use_bias=False, bias_init="zero", ndim=None, dtype=ms.float32,
                 mode='einsum'):
        super().__init__()
        self.mode = mode
        if isinstance(in_shape, int):
            in_shape = (in_shape,)
        if isinstance(out_shape, int):
            out_shape = (out_shape,)
        self.num_output_dims = len(out_shape)
        self.num_input_dims = len(in_shape)
        self.ndim = ndim
        self.dtype = dtype
        if self.ndim is None:
            self.ndim = len(in_shape) + 1
        if weight_init in ["relu", "linear"]:
            self.weight = custom_initializer(
                weight_init, in_shape + out_shape, dtype=ms.float32)
        else:
            self.weight = ms.Parameter(initializer.initializer(
                weight_init, in_shape + out_shape, dtype=ms.float32))
        self.use_bias = use_bias
        if self.use_bias is True:
            self.bias = ms.Parameter(
                initializer.initializer(bias_init, out_shape, dtype=ms.float32))
        if self.mode == 'ncon':
            ncon_list1 = [-i-1 for i in range(self.ndim - self.num_input_dims)] + [
                i+1 for i in range(len(in_shape))]
            ncon_list2 = (ncon_list1[self.ndim - self.num_input_dims:]) + \
                [-i-self.ndim+self.num_input_dims -
                    1 for i in range(len(out_shape))]
            self.ncon = Ncon([ncon_list1, ncon_list2])
        elif self.mode == 'einsum':
            in_letters = 'abcde'[: self.num_input_dims]
            out_letters = 'hijkl'[: self.num_output_dims]
            keep_letters = 'uvwxyz'[:(self.ndim - self.num_input_dims)]
            self.equation = f'{keep_letters}{in_letters}, {in_letters}{out_letters}->{keep_letters}{out_letters}'
            self.einsum = Einsum(self.equation)

    def construct(self, x):
        """CustomDense"""
        x = x.astype(self.dtype)
        if self.use_bias:
            if self.mode == 'ncon':
                output = self.ncon([x, self.weight.astype(
                    self.dtype)]) + self.bias.astype(self.dtype)
            else:
                output = self.einsum(x, self.weight.astype(
                    self.dtype)) + self.bias.astype(self.dtype)
        else:
            if self.mode == 'ncon':
                output = self.ncon([x, self.weight.astype(self.dtype)])
            else:
                output = self.einsum(x, self.weight.astype(self.dtype))
        return output


def custom_initializer(initializer_name, input_shape, dtype=ms.float32):
    """
    Custom initializer.
    Args:
        initializer_name (str): initializer name.
        input_shape (tuple): input shape.
        dtype (ms.dtype): dtype.
    Returns:
        Tensor: initialized tensor.
    """
    noise_scale = ms.Tensor(1.0)
    for channel_dim in input_shape:
        noise_scale /= channel_dim
    if initializer_name == 'relu':
        noise_scale *= 2
    stddev = ms.ops.sqrt(noise_scale)
    stddev = stddev / ms.Tensor(TRUNCATED_NORMAL_STDDEV_FACTOR)
    param = ms.Parameter(initializer.initializer(
        initializer.TruncatedNormal(stddev, 0), input_shape, dtype))
    return param
