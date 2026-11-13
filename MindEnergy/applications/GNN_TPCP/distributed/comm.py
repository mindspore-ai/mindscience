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


"""Collective primitives with hand-written backward, parameterized by axis.

Every helper is the identity when ``group.size == 1``: that is what lets one
code path be both "parallel" and "bit-identical to the serial baseline".
"""
from enum import Enum

from mindspore import _Function, mint
from mindspore.communication import comm_func


class GatherMode(Enum):
    """What the consumer of a gathered tensor holds.

    REDUCE: a shard        -> input gradient is a partial sum, backward must
                              reduce-scatter.
    SLICE:  a full replica -> every rank already computed the complete
                              gradient, backward must only slice.
    """
    REDUCE = "reduce"
    SLICE = "slice"

def _dim_of(x, g, dim):
    if x.ndim == 0:
        raise ValueError(f"cannot shard a 0-dim tensor along axis {dim}")

    if not -x.ndim <= dim < x.ndim:
        raise IndexError(f"axis {dim} out of range for tensor of shape {tuple(x.shape)}")

    dim = dim % x.ndim
    if x.shape[dim] % g.size:
        raise ValueError(f"axis {dim} of {tuple(x.shape)} not divisible by group size {g.size}")
    return dim

def all_reduce(x, g):
    if g.size == 1:
        return x
    out, _ = comm_func.all_reduce(x, group=g.name)
    return out

def all_gather(x, g, dim=0):
    if g.size == 1:
        return x
    dim = _dim_of(x, g, dim)
    y = mint.movedim(x, dim, 0).contiguous() if dim else x.contiguous()
    out, _ = comm_func.all_gather_into_tensor(y, group=g.name)
    return mint.movedim(out, 0, dim) if dim else out

def take_shard(x, g, dim=0):
    """Pick this rank's contiguous slice along ``dim`` (no communication)."""
    if g.size == 1:
        return x
    dim = _dim_of(x, g, dim)
    return mint.split(x, x.shape[dim] // g.size, dim)[g.rank].contiguous()

def reduce_scatter(x, g, dim=0):
    if g.size == 1:
        return x
    dim = _dim_of(x, g, dim)
    y = mint.movedim(x, dim, 0).contiguous() if dim else x.contiguous()
    out, _ = comm_func.reduce_scatter_tensor(y, group=g.name)
    return mint.movedim(out, 0, dim) if dim else out


class CopyToAll(_Function):
    """Input map of a column-parallel layer: identity forward, all-reduce back."""

    @staticmethod
    def forward(ctx, x, g):
        ctx.g = g
        return x

    @staticmethod
    def backward(ctx, grad):
        return all_reduce(grad, ctx.g), None


class Reduce(_Function):
    """Output map of a row-parallel layer: all-reduce forward, identity back."""

    @staticmethod
    def forward(ctx, x, g):
        return all_reduce(x, g)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class Gather(_Function):
    """all-gather forward along ``dim``; backward follows ``GatherMode``."""

    @staticmethod
    def forward(ctx, x, g, dim=0, mode=GatherMode.SLICE):
        ctx.g, ctx.dim, ctx.mode = g, dim, mode
        return all_gather(x, g, dim)

    @staticmethod
    def backward(ctx, grad):
        if ctx.mode is GatherMode.REDUCE:
            return reduce_scatter(grad, ctx.g, ctx.dim), None, None, None
        return take_shard(grad, ctx.g, ctx.dim), None, None, None


class ReduceScatter(_Function):
    """reduce + slice forward along ``dim``, all-gather back."""

    @staticmethod
    def forward(ctx, x, g, dim=0):
        ctx.g, ctx.dim = g, dim
        return reduce_scatter(x, g, dim)

    @staticmethod
    def backward(ctx, grad):
        return all_gather(grad, ctx.g, ctx.dim), None, None
