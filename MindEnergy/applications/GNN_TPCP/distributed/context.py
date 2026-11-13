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


"""Parallel context: orthogonal process groups for tp / cp / dp / pp.

Model-agnostic: knows nothing about tokens, heads, graphs or edges.
"""
from functools import reduce
from operator import mul

from mindspore import communication as comm
from mindspore.communication import (create_group, get_group_size,
                                     get_group_rank_from_world_rank, get_rank)

KINDS = ("tp", "cp", "dp", "dp-cp")
_GROUPS = {}

class Group:
    """Handle of one communication group. ``size == 1`` means "not parallelized"."""

    def __init__(self, kind, name=None, size=1):
        self.kind, self._name, self._size = kind, name, size

    @property
    def size(self):
        return self._size

    @property
    def rank(self):
        return 0 if self._name is None else get_group_rank_from_world_rank(get_rank(), self._name)

    @property
    def name(self):
        if self._name is None:
            raise RuntimeError(f"group '{self.kind}' not initialized; call init_parallel() first")
        return self._name

def _group_members(sizes, mask, world_size):
    """Split ranks into the orthogonal groups selected by ``mask``.

    Ranks sharing a coordinate on the selected axes fall into the same group,
    which is the Megatron layout: rank = sum(coord_i * prefix_prod(sizes)[i]).
    """
    stride = [1]
    for v in sizes:
        stride.append(stride[-1] * v)
    keep = [i for i, m in enumerate(mask) if m]
    keep_size = [sizes[i] for i in keep]
    keep_stride = [stride[i] for i in keep]
    gsize = reduce(mul, keep_size, 1)

    groups = []
    for g in range(world_size // gsize):
        base, rest = 0, g
        for i, m in enumerate(mask):
            if not m and sizes[i] > 1:
                base += (rest % sizes[i]) * stride[i]
                rest //= sizes[i]
        members = []
        for r in range(gsize):
            off, rest = 0, r
            for s, st in zip(keep_size, keep_stride):
                off += (rest % s) * st
                rest //= s
            members.append(base + off)
        groups.append(members)
    return groups

def init_parallel(tp=1, cp=1, pp=1, order="tp-cp-dp-pp"):
    """Create the orthogonal groups, after ``mindspore.communication.init()``."""
    if not comm._comm_helper._is_initialized():
        raise RuntimeError("mindspore communication is not initialized")
    world_size = get_group_size()
    if world_size % (tp * cp * pp):
        raise RuntimeError(f"world_size {world_size} not divisible by tp*cp*pp={tp * cp * pp}")
    axes = order.split("-")
    sizes = [{"tp": tp, "cp": cp, "dp": world_size // (tp * cp * pp), "pp": pp}[k] for k in axes]

    for kind in KINDS:
        mask = [k in kind.split("-") for k in axes]
        if reduce(mul, [s for s, m in zip(sizes, mask) if m], 1) == 1:
            _GROUPS[kind] = Group(kind)
            continue
        for members in _group_members(sizes, mask, world_size):
            if get_rank() in members:
                name = kind + "-" + "-".join(map(str, members))
                create_group(name, members)
                _GROUPS[kind] = Group(kind, name, len(members))
                break

def group(kind="tp"):
    """Group handle; degrades to a size-1 no-op group before init."""
    return _GROUPS.get(kind) or Group(kind)
