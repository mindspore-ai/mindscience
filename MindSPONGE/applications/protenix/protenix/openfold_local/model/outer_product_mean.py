# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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
2024-11-13 Modification:

- The 'bias' term has been removed in several linear layers.
- Use inhouse Layernorm in OuterProductMean to replace nn.Layernorm.

Protenix Team
"""
from dataclasses import dataclass
import numpy as np
from mindspore import nn, Parameter, mint
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from protenix.model import base_config
from mindscience.common.memory_reduce import _memory_reduce
from mindscience.common.initializer import lecun_init
from mindscience.models.layers.mask import MaskedLayerNorm
from mindscience.e3nn.utils import Ncon


class OuterProductMean(nn.Cell):
    """
    Implements the OuterProductMean operation for tensor computations.

    Args:
        config (Config): Configuration object containing parameters for the operation.
        global_config (GlobalConfig): Global configuration object.
        num_output_channel (int): Number of output channels.
        in_channel (int): Number of input channels.

    Inputs:
        - **act** (Tensor) - Input activation tensor.
        - **mask** (Tensor) - Mask tensor indicating valid regions in the input.

    Outputs:
        - **out** (Tensor) - Output tensor after applying the outer product mean operation.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        chunk_size: int = 128
        num_outer_channel: int = 32

    def __init__(self, config, global_config, num_output_channel, in_channel, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_output_channel = num_output_channel
        self.outer_product_mean = _OuterProductMean(self.config.num_outer_channel,
                                                    # self.config.chunk_size,
                                                    in_channel,
                                                    self.num_output_channel,
                                                    dtype=dtype)

    def construct(self, act, mask):
        if mask is None:
            mask = act.new_ones(act.shape[:-1])
        mask_norm = mint.unsqueeze(mint.matmul(mask.T, mask), -1)
        out = self.outer_product_mean(act, mask, mask_norm)
        return out


class _OuterProductMean(nn.Cell):
    r"""
    Computing the correlation of the input tensor along its second dimension, the computed correlation
    could be used to update the correlation features(e.g. the Pair representation).

    .. math::
        OuterProductMean(\mathbf{act}) = Linear(flatten(mean(\mathbf{act}\otimes\mathbf{act})))

    Args:
        num_outer_channel (float):  The last dimension size of intermediate layer in OuterProductMean.
        act_dim (int):              The last dimension size of the input act.
        num_output_channel (int):   The last dimension size of output.
        batch_size(int):            The batch size of parameters in OuterProductMean,
                                    used in while control flow. Default: "None".
        slice_num (int):            The slice num used in OuterProductMean layer
                                    when the memory is overflow. Default: 0.

    Inputs:
        - **act** (Tensor) - The input tensor with shape :math:`(dim_1, dim_2, act\_dim)`.
        - **mask** (Tensor) - The mask for OuterProductMean with shape :math:`(dim_1, dim_2)`.
        - **mask_norm** (Tensor) - Squared L2-norm along the first dimension of **mask**,
          pre-computed to avoid re-computing, its shape is :math:`(dim_2, dim_2, 1)`.
        - **index** (Tensor) - The index of while loop, only used in case of while control
          flow. Default: "None".

    Outputs:
        Tensor, the float tensor of the output of OuterProductMean layer with
        shape :math:`(dim_2, dim_2, num\_output\_channel)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import OuterProductMean
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> from mindspore.ops import operations as P
        >>> model = OuterProductMean(num_outer_channel=32, act_dim=128, num_output_channel=256)
        >>> act = Tensor(np.ones((32, 64, 128)), mstype.float32)
        >>> mask = Tensor(np.ones((32, 64)), mstype.float32)
        >>> mask_norm = P.ExpandDims()(P.MatMul(transpose_a=True)(mask, mask), -1)
        >>> output= model(act, mask, mask_norm)
        >>> print(output.shape)
        (64, 64, 256)
    """

    def __init__(self, num_outer_channel, act_dim, num_output_channel, batch_size=None, slice_num=0, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.num_output_channel = num_output_channel
        self.num_outer_channel = num_outer_channel
        self.layer_norm_input = MaskedLayerNorm()
        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.matmul = P.MatMul()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def construct(self, act, mask, mask_norm):
        """Compute outer product mean."""
        mask = mint.unsqueeze(mask, -1)
        act = self.layer_norm_input(
            act, self.layer_norm_input_gamma, self.layer_norm_input_beta)
        act_shape = act.shape
        if len(act_shape) != 2:
            act = act.reshape(-1, act_shape[-1])
        out_shape = act_shape[:-1] + (-1,)
        left_act = mask * mint.reshape(mint.matmul(act, self.left_projection_weight.astype(
            self.dtype).T) + self.left_projection_bias.astype(self.dtype), out_shape)

        right_act = mask * mint.reshape(mint.matmul(act, self.right_projection_weight.astype(
            self.dtype).T) + self.right_projection_bias.astype(self.dtype), out_shape)
        batched_inputs = (left_act,)
        nonbatched_inputs = (right_act, self.linear_output_weight.astype(self.dtype),
                             self.o_biases.astype(self.dtype))
        act = _memory_reduce(self._compute, batched_inputs,
                             nonbatched_inputs, self.slice_num, 1)
        epsilon = 1e-3
        # act = P.RealDiv()(act, epsilon + mask_norm)
        act = mint.div(act, epsilon + mask_norm)
        return act

    def _init_parameter(self):
        '''init parameter'''
        self.layer_norm_input_gamma = Parameter(
            Tensor(np.ones((self.act_dim)), ms.float32))
        self.layer_norm_input_beta = Parameter(
            Tensor(np.zeros((self.act_dim)), ms.float32))
        self.left_projection_weight = Parameter(
            initializer(lecun_init(self.act_dim), [self.num_outer_channel, self.act_dim], ms.float32))
        self.left_projection_bias = Tensor(
            np.zeros((self.num_outer_channel)), ms.float32)
        self.right_projection_weight = Parameter(
            initializer(lecun_init(self.act_dim), [self.num_outer_channel, self.act_dim], ms.float32))
        self.right_projection_bias = Tensor(
            np.zeros((self.num_outer_channel)), ms.float32)
        self.linear_output_weight = Parameter(
            Tensor(np.zeros((self.num_outer_channel, self.num_outer_channel, self.num_output_channel)),
                   ms.float32))
        self.o_biases = Parameter(
            Tensor(np.zeros((self.num_output_channel)), ms.float32))

    def _compute(self, left_act, right_act, linear_output_weight, linear_output_bias):
        '''compute outer product mean'''
        left_act = left_act.transpose((0, 2, 1))
        act = Ncon([[1, -2, -4], [1, -1, -3]])([left_act, right_act])
        act = Ncon([[-1, 1, 2, -2], [1, 2, -3]]
                   )([act, linear_output_weight.astype(self.dtype)]) + linear_output_bias.astype(self.dtype)
        act = act.transpose(1, 0, 2)
        return act
