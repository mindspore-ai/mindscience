# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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

"""primitives"""

from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, mint, Tensor
from protenix.model import base_config
from protenix.model.modules.module_utils.gated_linear_unit import gated_linear_unit
from protenix.model.modules import base_modules as bm


class AdaptiveLayernorm(nn.Cell):
    """
    If single condition is None, this layer is the same as layernorm.
    If single condition is given, the layer is modified from Scalable Diffusion Models with Transformers
    https://arxiv.org/abs/2212.09748

    Args:
        num_channels (int): Number of channels in the input tensor.
        single_channel (int, optional): Number of channels in the single condition tensor.
            Required if `with_single_cond` is True. Default: ``None``.
        with_single_cond (bool, optional): Whether to include the single condition adaptation. Default: ``True``.

    Inputs:
        - **x** (Tensor) - Input tensor to be normalized.
        - **single_cond** (Tensor, optional) - Optional single condition tensor
        used to adapt the normalization parameters.
            Required if `with_single_cond` is True.

    Outputs:
        - **output** (Tensor) - The normalized output tensor.
    """

    def __init__(self, num_channels, single_channel=None, with_single_cond=True, dtype=ms.float32):
        super().__init__()
        self.with_single_cond = with_single_cond
        if self.with_single_cond:
            self.layernorm = bm.LayerNorm([num_channels], name='layer_norm',
                                          create_gamma=False, create_beta=False,
                                          gamma_init='ones', beta_init='zeros', dtype=ms.float32)
            self.single_cond_layer_norm = bm.LayerNorm([single_channel], name='single_cond_layer_norm',
                                                       create_beta=False, gamma_init='ones', beta_init='zeros',
                                                       dtype=ms.float32)
            self.single_cond_scale = bm.LinearAMP(single_channel, num_channels, weight_init='zeros',
                                                  has_bias=True, bias_init='ones', dtype=dtype)
            self.single_cond_bias = bm.LinearAMP(
                single_channel, num_channels, weight_init='zeros', has_bias=False, dtype=dtype)
        else:
            self.layernorm = bm.LayerNorm([num_channels], dtype=ms.float32)

    def construct(self, x, single_cond=None):
        """construct"""
        if not self.with_single_cond:
            x = self.layernorm(x)
        else:
            x = self.layernorm(x)
            single_cond = self.single_cond_layer_norm(single_cond)
            single_scale = self.single_cond_scale(single_cond)
            single_bias = self.single_cond_bias(single_cond)
            x = mint.add(mint.mul(mint.sigmoid(single_scale.astype(
                ms.float32)).astype(x.dtype), x), single_bias)
        return x


class AdaptiveZeroInit(nn.Cell):
    """
    An adaptive initialization layer that combines two conditional linear transformations.

    Args:
        global_config: Configuration object containing initialization settings.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        single_channels (int, optional): Number of single conditional channels. Default: ``None``.
        ndim (int, optional): Number of dimensions for the dense layer input. Default: ``3``.
        with_single_cond (bool, optional): Whether to use single conditional transformation. Default: ``True``.

    Inputs:
        - **x** (Tensor) - Input tensor to the layer.
        - **single_cond** (Tensor, optional) - Single conditional tensor. Required if `with_single_cond` is True.

    Outputs:
        - **output** (Tensor) - Output tensor after applying the adaptive initialization.
    """

    def __init__(self, in_channels, out_channels, single_channels=None, with_single_cond=True, dtype=ms.float32):
        super().__init__()
        self.with_single_cond = with_single_cond
        self.cond_linear1 = bm.LinearAMP(
            in_channels, out_channels, weight_init='ones', has_bias=False, dtype=dtype)
        if self.with_single_cond:
            if single_channels is None:
                single_channels = in_channels
            self.cond_linear2 = bm.LinearAMP(single_channels, out_channels, weight_init='zeros',
                                             has_bias=True, bias_init='zeros', dtype=dtype)
            self.cond_linear2.linear.bias = ms.Parameter(
                self.cond_linear2.linear.bias * (-2))

    def construct(self, x, single_cond=None):
        if not self.with_single_cond:
            output = self.cond_linear1(x)
        else:
            output = self.cond_linear1(x)
            cond = self.cond_linear2(single_cond)
            output = mint.mul(mint.sigmoid(cond.astype(
                ms.float32)).astype(cond.dtype), output)
        return output


class TransitionBlock(nn.Cell):
    """
    A transition block for transformer networks, implementing either a GLU-based or linear-based transformation.

    Args:
        config (Config): Configuration object containing parameters for the transition block.
        global_config (GlobalConfig): Global configuration object.
        normalized_shape (tuple): Shape of the input tensor for normalization.
        ndim (int): Number of dimensions of the input tensor. Default: ``3``.

    Inputs:
        - **act** (Tensor) - Input activation tensor to be processed.

    Outputs:
        - **output** (Tensor) - Output tensor after processing through the transition block.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        num_intermediate_factor: int = 4
        use_glu_kernel: bool = False

    def __init__(
        self, config, global_config, normalized_shape, ndim=3, dtype=ms.float32
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        num_channels = normalized_shape[-1]
        self.num_intermediate = int(
            num_channels * self.config.num_intermediate_factor)
        self.layernorm = bm.LayerNorm(
            normalized_shape, name='input_layer_norm', dtype=ms.float32)
        if self.config.use_glu_kernel:
            self.glu_weight = bm.custom_initializer(
                'relu', (num_channels, 2 * self.num_intermediate), dtype=ms.float32)
            self.glu_weight = ms.Parameter(Tensor(self.glu_weight).reshape(
                num_channels, 2, self.num_intermediate))
        else:
            self.linear1 = bm.CustomDense(
                num_channels, self.num_intermediate, weight_init='ones', ndim=ndim, dtype=dtype)
            self.linear2 = bm.CustomDense(
                num_channels, self.num_intermediate, weight_init='ones', ndim=ndim, dtype=dtype)
        self.out_linear = bm.CustomDense(self.num_intermediate, num_channels,
                                         weight_init=self.global_config.final_init, ndim=ndim, dtype=dtype)

    def construct(self, act):
        """construct"""
        act = self.layernorm(act)
        if self.config.use_glu_kernel:
            c = gated_linear_unit(
                x=act,
                weight=self.glu_weight,
                activation=mint.nn.functional.silu,
            )
        else:
            a = self.linear1(act)
            b = self.linear2(act)
            c = mint.nn.functional.silu(a) * b
        return self.out_linear(c)
