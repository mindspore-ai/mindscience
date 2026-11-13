# Copyright 2026 Huawei Technologies Co., Ltd
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







"""Parallel linear layers plus the parameter tag registry.

The registry is the model-agnostic replacement for name-substring heuristics
("lin_node", "att_", ...): a parameter carries its own tag, so no layer ever
has to know how the host model named it.
"""



import math

import mindspore as ms
from mindspore import nn, mint, ops, Parameter, _Function
from mindspore.common.initializer import initializer, Uniform

from .comm import CopyToAll, Gather, GatherMode, Reduce, ReduceScatter, all_gather, take_shard
from .context import group as _group

SHARDED, COMPLETE = "sharded", "complete"
_REGISTRY = {}  # id(param) -> (tag, param); the strong ref prevents id reuse

def _tag(param, tag):
    _REGISTRY[id(param)] = (tag, param)
    return param

def mark_sharded(param):
    """Parameter holds only 1/size of the global weight."""
    return _tag(param, SHARDED)

def mark_grad_complete(param):
    """Replicated parameter whose local gradient is already the global one."""
    return _tag(param, COMPLETE)

def tag_of(param):
    entry = _REGISTRY.get(id(param))
    return entry[0] if entry else None

def is_sharded(param):
    return tag_of(param) == SHARDED

def is_grad_complete(param):
    return tag_of(param) == COMPLETE

class Shard(_Function):
    """Slice the input along ``dim``; backward all-gathers (dual of ``Gather``)."""

    @staticmethod
    def forward(ctx, x, g, dim=0):
        ctx.g, ctx.dim = g, dim
        return take_shard(x, g, dim)

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad, ctx.g, ctx.dim), None, None

def initialize_affine_weight(shape, g=None, partition_dim=0, init_method="XavierUniform",
                             init_dtype=ms.float32, num_groups=None):
    """Build a parameter from the *global* shape, then keep this rank's slice.

    Initializing from the global shape is what keeps ``size=N`` weights equal to
    a slice of the ``size=1`` weight, i.e. a run at size=N starts from exactly
    the same point as the serial baseline.

    ``num_groups`` guards grouped splits: the axis must be cut on a group
    boundary, otherwise a shard would mix two logical groups (e.g. two heads).
    Divisibility alone does not imply that.
    """
    g = g or _group("tp")
    param = Parameter(initializer(init_method, shape, init_dtype))
    if g.size == 1:
        return param
    length = shape[partition_dim]
    if length % g.size:
        raise ValueError(f"axis {partition_dim} (len {length}) not divisible by {g.size}")
    if num_groups is not None and num_groups % g.size:
        raise ValueError(
            f"{num_groups} groups cannot be spread over {g.size} shards: a shard would "
            f"start and end inside a group, so grouped layouts (e.g. heads) break silently")
    shard = mint.split(param, length // g.size, partition_dim)[g.rank]
    return mark_sharded(Parameter(shard.contiguous()))

class ColumnParallelLinear(nn.Cell):
    """Split ``out_features`` across the group; optionally consume sharded input."""

    def __init__(self, in_features, out_features, bias=False, gather_output=True,
                 use_sequence_parallel=False, seq_dim=0, num_groups=None,
                 init_method="XavierUniform", init_dtype=ms.float32,
                 compute_dtype=ms.float32, weight_init=None):
        super().__init__()
        self.g = _group("tp")
        self.seq_dim, self.compute_dtype = seq_dim, compute_dtype
        self.num_groups = num_groups
        if self.g.size == 1:
            # Passthrough: identical module, identical kernel, identical numbers.
            self.layer = mint.nn.Linear(in_features, out_features, bias=bias,
                                        weight_init=weight_init or init_method)
            return
        self.layer = None
        self.in_map = Gather(dim=seq_dim, mode=GatherMode.REDUCE) if use_sequence_parallel else CopyToAll
        self.out_map = Gather(dim=-1, mode=GatherMode.SLICE) if gather_output else None
        self.weight = initialize_affine_weight((in_features, out_features), self.g, 1,
                                               init_method, init_dtype, num_groups)
        self.bias = initialize_affine_weight(
            (out_features,), self.g, 0, Uniform(1 / math.sqrt(in_features)), init_dtype,
            num_groups) if bias else None

    def construct(self, x):
        if self.layer is not None:
            return self.layer(x)
        x = self.in_map(x, self.g) if self.in_map is not None else x
        x = mint.matmul(x.astype(self.compute_dtype), ops.cast(self.weight, self.compute_dtype))
        if self.bias is not None:
            x = x + ops.cast(self.bias, self.compute_dtype)
        return self.out_map(x, self.g) if self.out_map is not None else x

class RowParallelLinear(nn.Cell):
    """Split ``in_features`` across the group; optional sequence-parallel output."""

    def __init__(self, in_features, out_features, bias=False, input_is_parallel=False,
                 use_sequence_parallel=False, seq_dim=0, num_groups=None,
                 init_method="XavierUniform", init_dtype=ms.float32,
                 compute_dtype=ms.float32, weight_init=None):
        super().__init__()
        self.g = _group("tp")
        self.seq_dim, self.compute_dtype = seq_dim, compute_dtype
        self.num_groups = num_groups
        if self.g.size == 1:
            self.layer = mint.nn.Linear(in_features, out_features, bias=bias,
                                        weight_init=weight_init or init_method)
            return
        self.layer = None
        self.in_map = None if input_is_parallel else Shard(dim=-1)
        self.out_map = ReduceScatter(dim=seq_dim) if use_sequence_parallel else Reduce
        self.weight = initialize_affine_weight((in_features, out_features), self.g, 0,
                                               init_method, init_dtype, num_groups)
        # Bias lives *after* the reduction, so its local gradient is already
        # complete; adding it before the reduce would scale it by size.
        self.bias = mark_grad_complete(Parameter(initializer(
            Uniform(1 / math.sqrt(in_features)), out_features, init_dtype))) if bias else None

    def construct(self, x):
        if self.layer is not None:
            return self.layer(x)
        x = self.in_map(x, self.g) if self.in_map is not None else x
        x = mint.matmul(x.astype(self.compute_dtype), ops.cast(self.weight, self.compute_dtype))
        x = self.out_map(x, self.g)
        if self.bias is not None:
            x = x + ops.cast(self.bias, self.compute_dtype)
        return x
