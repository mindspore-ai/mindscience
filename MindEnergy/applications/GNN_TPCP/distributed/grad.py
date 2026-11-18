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


"""Gradient synchronisation plan for tensor parallelism.

Three buckets, and the two questions they answer are *not* the same:

    bucket    parameter        local gradient   all-reduce?   norm reduce?
    sharded   1/size slice     already global   no            yes
    complete  replica          already global   no            no
    partial   replica          partial sum      yes           no

Confusing  "gradient is a partial sum"  with  "parameter is sharded"  is the
single most common source of silent ``size``-fold errors, so the classification
is materialised once, in an explicit plan object, instead of being re-derived
at every call site.
"""



import math
from dataclasses import dataclass, field

import mindspore as ms
from mindspore import mint

from .layers import is_grad_complete, is_sharded




@dataclass
class ReducePlan:
    partial: list = field(default_factory=list)   # need a cross-group SUM
    complete: list = field(default_factory=list)  # already global, skip
    sharded: list = field(default_factory=list)   # already global, but norm needs SUM
    buckets: list = field(default_factory=list)   # dtype-fused groups of `partial`


def _build_buckets(params, idx):
    by_dtype = {}
    for i in idx:
        by_dtype.setdefault(str(params[i].dtype), []).append(i)
    buckets = []
    for dtype in sorted(by_dtype):
        ids = by_dtype[dtype]
        sizes = [math.prod(params[i].shape) for i in ids]
        offsets, acc = [], 0
        for n in sizes:
            offsets.append(acc)
            acc += n
        buckets.append({"dtype": dtype, "idx": ids, "sizes": sizes, "offsets": offsets,
                        "shapes": [tuple(params[i].shape) for i in ids], "total": acc})
    return buckets



def build_plan(params, fuse=True, extra_partial=()):
    """Classify every trainable parameter once, before the training loop.

    ``extra_partial`` lets the host force parameters into the ``partial`` bucket
    when it knows better than the registry (e.g. a layer it built by hand).
    """
    plan = ReducePlan()
    for i, p in enumerate(params):
        if is_sharded(p):
            plan.sharded.append(i)
        elif is_grad_complete(p):
            plan.complete.append(i)
        else:
            plan.partial.append(i)
    plan.partial.extend(i for i in extra_partial if i not in plan.partial)
    if fuse:
        plan.buckets = _build_buckets(params, plan.partial)
    return plan



def reduce_grads(grads, plan, all_reduce):
    """Sum the partial-sum gradients across the group, fused per dtype.

    Fusing is element-wise exact: each element is still summed exactly once,
    so the result is bit-identical to one all-reduce per parameter.
    """
    if not plan.partial:
        return grads
    if plan.buckets:
        for b in plan.buckets:
            flat = mint.cat([grads[i].reshape(-1) for i in b["idx"]])
            flat = all_reduce(flat)
            for i, shape, n, off in zip(b["idx"], b["shapes"], b["sizes"], b["offsets"]):
                grads[i].copy_(flat[off:off + n].reshape(shape))
        return grads
    for i in plan.partial:
        grads[i].copy_(all_reduce(grads[i]))
    return grads


def global_norm(grads, plan=None, all_reduce=None):
    """L2 norm over all parameters.

    Only ``sharded`` parameters contribute a *partial* square sum, so only that
    part is reduced. Reducing the replicated ones would inflate the norm by
    ``size`` and silently shrink the effective clip coefficient.

    Raises:
        ValueError: When ``plan`` classifies parameters as ``sharded`` but no
            ``all_reduce`` is supplied. Returning the unreduced norm would be
            exactly the silent ``size``-fold drift this module exists to
            prevent, so it fails loudly instead.
    """
    squares = [(g.astype(ms.float32) ** 2).sum() for g in grads]
    total = mint.stack(squares).sum()
    if plan is None:
        return mint.sqrt(total)
    if plan.sharded and all_reduce is None:
        raise ValueError(
            "plan has sharded parameters but all_reduce is None: the norm would come out "
            "silently off by the shard factor; pass an all_reduce for the sharded bucket "
            "or a plan without sharded parameters")
    if not plan.sharded:
        return mint.sqrt(total)
    shard_sq = mint.stack([squares[i] for i in plan.sharded]).sum()
    return mint.sqrt(total - shard_sq + all_reduce(shard_sq))
