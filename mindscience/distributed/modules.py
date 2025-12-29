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
"""
Distributed model-parallel modules and helpers.

Provides parallel linear layers and weight initialization utilities used with
distributed mappings and communication groups.
"""


import math
from typing import Union, Tuple, Optional

import mindspore as ms
from mindspore import nn, ops, mint, Parameter
from mindspore.common.initializer import initializer, Initializer, Uniform, HeUniform

from mindscience.distributed.mappings import (
    CopyToAll, GatherFromHidden,
    ScatterToHidden, ReduceFromAll,
    GatherFromSequence, ReduceScatterToSequence,
)
from mindscience.distributed.manager import get_tensor_parallel_rank, get_tensor_parallel_group


class Identity(nn.Cell):
    def construct(self, x, _):
        return x


def initialize_affine_weight(
    init_shape: Tuple[int],
    tp_world_size: int,
    partition_dim: int,
    init_method: Union[Initializer, str] = "XavierUniform",
    init_dtype = ms.float32
):
    """Initialize and (optionally) partition a weight tensor for parallelism.

    Args:
        init_shape (Tuple[int]): Shape of the weight tensor to initialize.
        tp_world_size (int): Tensor parallel world size.
        partition_dim (int): Dimension along which to partition the weight tensor.
        init_method (Union[Initializer, str], optional): Initialization method to use.
            Default: ``"XavierUniform"``.
        init_dtype (mstype.dtype, optional): Data type for the initialized weight. Default: ``ms.float32``.

    Returns:
        Parameter, The full parameter when tp_world_size==1 or the per-rank partition otherwise.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindscience.distributed import initialize_parallel
        >>> from mindscience.distributed.modules import initialize_affine_weight
        >>> init()
        >>> initialize_parallel(tensor_parallel_size=2)
        >>> param = initialize_affine_weight((64, 64), 2, 0)
        >>> print(param.shape)
        (32, 64)
    """
    master_weight = Parameter(initializer(init_method, init_shape, init_dtype))
    if tp_world_size == 1:
        return master_weight
    per_partition_size = init_shape[partition_dim] // tp_world_size
    weight_list = mint.split(master_weight, per_partition_size, partition_dim)
    tp_rank = get_tensor_parallel_rank()
    return Parameter(weight_list[tp_rank])


class ColumnParallelLinear(nn.Cell):
    """Column-parallel linear layer that shards the output feature dimension across TP ranks.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features across all TP ranks.
        bias (bool, optional): Whether create and partition a bias parameter consistent with the output sharding.
                               Default: ``True``.
        gather_output (bool, optional): Whether gather the TP-local outputs into a full output via.
                                        Default: ``True``.
        use_sequence_parallel (bool, optional): Whether use all gather for input instead of broadcast.
                                                Default: ``False``.
        weight_init (Union[Initializer, str], optional): Weight initialization method. Default: ``None``.
        bias_init (Union[Initializer, str], optional): Bias initialization method. Default: ``None``.
        param_init_dtype (mstype.dtype, optional): Parameter initialization data type. Default: ``ms.float32``.
        compute_dtype (mstype.dtype, optional): Computation data type. Default: ``ms.bfloat16``.

    Inputs:
        - **x** (Tensor): Input tensor of shape (seq_len // TP, in_features) or (seq_len, in_features),
          depending on whether `use_sequence_parallel` is True.

    Outputs:
        - **output** (Tensor): Output tensor of shape (seq_len, out_features) or (seq_len, out_features // TP),
          depending on whether `gather_output` is True.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindscience.distributed import initialize_parallel
        >>> from mindscience.distributed.modules import ColumnParallelLinear
        >>> init()
        >>> initialize_parallel(tensor_parallel_size=2)
        >>> linear = ColumnParallelLinear(in_features=512, out_features=1024, bias=True)
        >>> input_tensor = ms.mint.randn(32, 512)
        >>> output = linear(input_tensor)
        >>> print(output.shape)
        (32, 1024)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        use_sequence_parallel: bool = False,
        weight_init: Optional[Union[Initializer, str]] = None,
        bias_init: Optional[Union[Initializer, str]] = None,
        param_init_dtype = ms.float32,
        compute_dtype = ms.bfloat16
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.tp_group = get_tensor_parallel_group()

        self.in_map = GatherFromSequence.apply if use_sequence_parallel else CopyToAll.apply
        self.out_map = GatherFromHidden.apply if gather_output else Identity()

        assert (
            out_features % self.tp_group.size == 0
        ), f"ColumnParallelLinear out_features {out_features} is not divisible by tp_world_size {self.tp_group.size}."

        if weight_init is None:
            weight_init = HeUniform(math.sqrt(5))
        self.weight = initialize_affine_weight(
            (out_features, in_features), self.tp_group.size, 0, weight_init, param_init_dtype
        )

        self.bias = None
        if bias:
            if bias_init is None:
                bias_init = Uniform(1 / math.sqrt(in_features))
            self.bias = initialize_affine_weight((out_features,), self.tp_group.size, 0, bias_init, param_init_dtype)

    def construct(self, x):
        x = self.in_map(x, self.tp_group)
        weight = ops.cast(self.weight, self.compute_dtype)
        x = mint.matmul(x, weight.T)
        if self.bias is not None:
            bias = ops.cast(self.bias, self.compute_dtype)
            x = x + bias
        x = self.out_map(x, self.tp_group)
        return x


class RowParallelLinear(nn.Cell):
    """Row-parallel linear layer that shards the input feature dimension across TP ranks.

    Args:
        in_features (int): Total number of input features across all TP ranks.
        out_features (int): Number of output features.
        bias (bool, optional): Whether create a bias parameter (not sharded along input dim).
                               Default: ``True``.
        input_is_parallel (bool, optional): Whether expect the input already partitioned
                                            across TP ranks. Default: ``False``.
        use_sequence_parallel (bool, optional): Whether use reduce scatter for output instead of all reduce.
                                                Default: ``False``.
        weight_init (Union[Initializer, str], optional): Weight initialization method. Default: ``None``.
        bias_init (Union[Initializer, str], optional): Bias initialization method. Default: ``None``.
        param_init_dtype (mstype.dtype, optional): Parameter initialization data type. Default: ``ms.float32``.
        compute_dtype (mstype.dtype, optional): Computation data type. Default: ``ms.bfloat16``.

    Inputs:
        - **x** (Tensor): Input tensor of shape (seq_len, in_features // TP) or (seq_len, in_features),
          depending on whether `input_is_parallel` is True.

    Outputs:
        - **output** (Tensor): Output tensor of shape (seq_len // TP, out_features) or (seq_len, out_features),
          depending on whether `use_sequence_parallel` is True.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindscience.distributed import initialize_parallel
        >>> from mindscience.distributed.modules import RowParallelLinear
        >>> init()
        >>> initialize_parallel(tensor_parallel_size=2)
        >>> linear = RowParallelLinear(in_features=1024, out_features=512, bias=True)
        >>> input_tensor = ms.mint.randn(32, 1024)
        >>> output = linear(input_tensor)
        >>> print(output.shape)
        (32, 512)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        use_sequence_parallel: bool = False,
        weight_init: Optional[Union[Initializer, str]] = None,
        bias_init: Optional[Union[Initializer, str]] = None,
        param_init_dtype = ms.float32,
        compute_dtype = ms.bfloat16
    ):
        super().__init__()
        self.compute_dtype= compute_dtype
        self.tp_group = get_tensor_parallel_group()

        self.in_map = Identity() if input_is_parallel else ScatterToHidden.apply
        self.out_map = ReduceScatterToSequence.apply if use_sequence_parallel else ReduceFromAll.apply

        assert (
            in_features % self.tp_group.size == 0
        ), f"RowParllelLinear in_features {in_features} is not divisible by tp_world_size {self.tp_group.size}."

        if weight_init is None:
            weight_init = HeUniform(math.sqrt(5))
        self.weight = initialize_affine_weight(
            (out_features, in_features), self.tp_group.size, 1, weight_init, param_init_dtype
        )

        self.bias = None
        if bias:
            if bias_init is None:
                bias_init = Uniform(1 / math.sqrt(in_features))
            self.bias = Parameter(initializer(bias_init, out_features, param_init_dtype))

    def construct(self, x):
        x = self.in_map(x, self.tp_group)
        weight = ops.cast(self.weight, self.compute_dtype)
        x = mint.matmul(x, weight.T)
        x = self.out_map(x, self.tp_group)
        if self.bias is not None:
            bias = ops.cast(self.bias, self.compute_dtype)
            x = x + bias
        return x
