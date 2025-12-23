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

"""Triangle"""
from typing import Literal
from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, mint

from protenix.model import base_config
from protenix.model.modules import base_modules as bm
from protenix.model.modules.base_modules import LayerNorm, CustomDense


class _TriangleMultiplication(nn.Cell):
    r"""
    Triangle multiplication layer. for the detailed implementation process, refer to
    `TriangleMultiplication <https://www.nature.com/articles/s41586-021-03819-2>`_.

    The information between the amino acid pair is integrated through the information of three edges ij, ik, jk, and
    the result of the dot product between ik and jk is added to the edge of ij.

    Args:
        num_intermediate_channel (float):   The number of intermediate channel.
        equation (str):                     The equation used in triangle multiplication layer. edge update forms
                                            corresponding to 'incoming' and 'outgoing',
                                            :math:`(ikc,jkc->ijc, kjc,kic->ijc)`.
        layer_norm_dim (int):               The last dimension length of the layer norm.
        batch_size (int):                   The batch size of parameters in triangle multiplication. Default: ``None``.

    Inputs:
        - **pair_act** (Tensor) - Tensor of pair_act. shape :math:`(N{res}, N{res}, layer\_norm\_dim)`.
        - **pair_mask** (Tensor) - The mask for TriangleAttention matrix with shape. shape :math:`(N{res}, N{res})`.
        - **index** (Tensor) - The index of while loop, only used in case of while control
          flow.

    Outputs:
        Tensor, the float tensor of the pair_act of the layer with shape :math:`(N{res}, N{res}, layer\_norm\_dim)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import TriangleMultiplication
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> model = TriangleMultiplication(num_intermediate_channel=64,
        ...                                equation="ikc,jkc->ijc", layer_norm_dim=64, batch_size=0)
        >>> input_0 = Tensor(np.ones((256, 256, 64)), mstype.float32)
        >>> input_1 = Tensor(np.ones((256, 256)), mstype.float32)
        >>> out = model(input_0, input_1, index=0)
        >>> print(out.shape)
        (256, 256, 64)
    """

    def __init__(self, config, global_config, num_intermediate_channel,
                 normalized_shape, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_intermediate_channel = num_intermediate_channel
        self.left_norm_input = LayerNorm(
            normalized_shape, create_beta=True, dtype=ms.float32)
        self.center_norm = LayerNorm(
            normalized_shape, create_beta=True, dtype=ms.float32)
        self.output_projection = CustomDense(
            normalized_shape[-1], num_intermediate_channel,
            weight_init=self.global_config.final_init,
            ndim=3, dtype=dtype)
        self.gating_linear = CustomDense(
            num_intermediate_channel, num_intermediate_channel,
            weight_init=self.global_config.final_init, ndim=3, dtype=dtype)
        if self.config.equation == "ikc,jkc->ijc":
            self.out = True
        elif self.config.equation == "kjc,kic->ijc":
            self.out = False
        else:
            raise ValueError("Not support this equation.")

        # add for protenix
        self.linear_ga = bm.LinearAMP(
            num_intermediate_channel, num_intermediate_channel, has_bias=False, dtype=dtype)
        self.linear_pa = bm.LinearAMP(
            num_intermediate_channel, num_intermediate_channel, has_bias=False, dtype=dtype)
        self.linear_gb = bm.LinearAMP(
            num_intermediate_channel, num_intermediate_channel, has_bias=False, dtype=dtype)
        self.linear_pb = bm.LinearAMP(
            num_intermediate_channel, num_intermediate_channel, has_bias=False, dtype=dtype)

    def construct(self, act, mask=None):
        r"""
        Builds triangle multiplication module.

        Args:
            act(Tensor):     Pair activations. Data type is float.
            mask(Tensor):    Pair mask. Data type is float.

        Returns:
            act(Tensor), the shape is same as act_shape[:-1].
        """
        input_act = act

        if mask is None:
            mask = act.new_ones(act.shape[:-1])
        mask = mask[None, ...]
        act = self.left_norm_input(act)

        a = ms.mint.sigmoid(self.linear_ga(act))
        a = a * self.linear_pa(act)
        a = a.transpose(2, 0, 1) * mask
        b = ms.mint.sigmoid(self.linear_gb(act))
        b = b * self.linear_pb(act)
        b = b.transpose(2, 0, 1) * mask

        if not self.out:
            a = a.transpose(-1, -2)
        else:
            b = b.transpose(-1, -2)
        act = mint.matmul(a, b)
        act = self.center_norm(act.transpose((1, 2, 0)))
        act = self.output_projection(act)
        gate_out = self.gating_linear(input_act)
        act = act * mint.sigmoid(gate_out.astype(ms.float32)
                                 ).astype(gate_out.dtype)
        return act


class TriangleMultiplication(nn.Cell):
    """
    Implements triangle multiplication for tensor operations.

    Args:
        config (Config): Configuration object specifying the equation and whether to use a GLU kernel.
        global_config (GlobalConfig): Global configuration object.
        in_channel (int): Number of input channels.
        normalized_shape (tuple): Shape of the input tensor for normalization.
        batch_size (int, optional): Batch size for processing. Default: ``None``.

    Inputs:
        - **act** (Tensor) - Input activation tensor.
        - **mask** (Tensor) - Mask tensor indicating valid regions in the input.

    Outputs:
        - **out** (Tensor) - Output tensor after triangle multiplication.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        equation: Literal['ikc,jkc->ijc', 'kjc,kic->ijc']
        use_glu_kernel: bool = False

    def __init__(self, config, global_config, in_channel, normalized_shape, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.triangle_multi = _TriangleMultiplication(
            self.config,
            self.global_config,
            num_intermediate_channel=in_channel,
            normalized_shape=normalized_shape,
            dtype=dtype)

    def construct(self, act, mask):
        out = self.triangle_multi(act, mask)
        return out
