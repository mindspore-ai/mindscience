# Copyright 2024 DeepMind Technologies Limited
# Copyright (C) 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications by Huawei Technologies Co., Ltd: Adapt to run by MindSpore on Ascend

"""Common modules."""

from collections.abc import Sequence
import contextlib
import numbers
from typing import TypeAlias

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common import initializer
from mindscience.e3nn.utils import Ncon

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

    def __init__(self, normalized_shape, name=None, begin_norm_axis=-1,
                 begin_params_axis=-1, gamma_init='ones',
                 beta_init='zeros', epsilon=1e-5, dtype=ms.float32,
                 create_beta=True, create_gamma=True):
        super().__init__()
        if not create_beta:
            beta_init = 'zeros'
        if not create_gamma:
            gamma_init = 'ones'
        self.layernorm = nn.LayerNorm(normalized_shape[begin_norm_axis:], begin_norm_axis=begin_norm_axis,
                                      begin_params_axis=begin_params_axis, gamma_init=gamma_init,
                                      beta_init=beta_init, epsilon=epsilon, dtype=dtype)
        if create_beta is False:
            self.layernorm.beta.requires_grad = False
        if create_gamma is False:
            self.layernorm.gamma.requires_grad = False
        self.dtype = dtype

    def construct(self, x):
        out = self.layernorm(x.astype(ms.float32)).astype(x.dtype)
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

    def __init__(self, in_shape, out_shape, weight_init="zeros", use_bias=False, \
                 bias_init="zeros", ndim=None, dtype=ms.float32):
        super().__init__()
        if isinstance(in_shape, int):
            in_shape = (in_shape,)
        if isinstance(out_shape, int):
            out_shape = (out_shape,)
        self.num_output_dims = len(out_shape)
        self.num_input_dims = len(in_shape)
        if ndim is None:
            ndim = len(in_shape) + 1
        if weight_init in ["relu", "linear"]:
            self.weight = custom_initializer(
                weight_init, in_shape + out_shape, dtype=dtype)
        else:
            self.weight = ms.Parameter(initializer.initializer(
                weight_init, in_shape + out_shape, dtype=dtype))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = ms.Parameter(
                initializer.initializer(bias_init, out_shape, dtype=dtype))
        ncon_list1 = [-i-1 for i in range(ndim - self.num_input_dims)] + [
            i+1 for i in range(len(in_shape))]
        ncon_list2 = (ncon_list1[ndim - self.num_input_dims:]) + \
            [-i-ndim+self.num_input_dims-1 for i in range(len(out_shape))]
        self.ncon = Ncon([ncon_list1, ncon_list2])

        in_letters = 'abcde'[: self.num_input_dims]
        out_letters = 'hijkl'[: self.num_output_dims]
        self.equation = f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'

    def construct(self, x):
        if self.use_bias:
            output = self.ncon([x, self.weight]) + self.bias
        else:
            output = self.ncon([x, self.weight])
        return output


def custom_initializer(initializer_name, input_shape, dtype=ms.float32):
    """custom initializer"""
    noise_scale = ms.Tensor(1.0)
    for channel_dim in input_shape:
        noise_scale /= channel_dim
    if initializer_name == 'relu':
        noise_scale *= 2
    stddev = ops.sqrt(noise_scale)
    stddev = stddev / ms.Tensor(TRUNCATED_NORMAL_STDDEV_FACTOR)
    param = ms.Parameter(initializer.initializer(
        initializer.TruncatedNormal(stddev, 0), input_shape, dtype))
    return param
