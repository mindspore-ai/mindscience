
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

"""chunk"""

from mindspore import nn, ops, Tensor


def _slice_along_axis(x: Tensor, start: int, size: int, axis: int) -> Tensor:
    rank = len(x.shape)
    if axis < 0:
        axis = axis + rank
    begin = [0] * rank
    begin[axis] = start
    sizes = list(x.shape)
    sizes[axis] = size
    return ops.slice(x, tuple(begin), tuple(sizes))


def _tree_concat(chunks, axis: int):
    """把一批分块输出按 axis 拼回（支持嵌套结构）。"""
    ref = chunks[0]
    if isinstance(ref, Tensor):
        return ops.concat(chunks, axis=axis)
    if isinstance(ref, (list, tuple)):
        # 对每个位置递归拼接
        cols = list(zip(*chunks))
        merged = [_tree_concat(list(col), axis) for col in cols]
        return type(ref)(merged)
    if isinstance(ref, dict):
        keys = ref.keys()
        return {k: _tree_concat([c[k] for c in chunks], axis) for k in keys}
    # 标量/None/其他不可拼接对象：直接返回第一个
    return ref


def apply_in_chunks_multi(
    cell: nn.Cell,
    *args,
    chunk_size: int,
    chunk_axes,  # type: list | tuple  # 与 args 等长；对应 arg 的分片维度，None 表示不分片
    out_axis: int,                        # 输出沿哪个维度拼回（通常是主序列维）
    kw_axes=None,  # type: dict | None  # {kw_name: axis or None}
    **kwargs
):
    """
    示例：
      y = apply_in_chunks_multi(attn, q, k, v, mask,
                                chunk_size=64,
                                chunk_axes=[1, 1, 1, 3],  # q,k,v按T维切；mask按最后一维切
                                out_axis=1)
    """
    if kw_axes is None:
        kw_axes = {}
    if len(chunk_axes) != len(args):
        raise ValueError(f"chunk_axes length {len(chunk_axes)} is not equal to args length {len(args)}")

    # 选一个“主切分长度”做基准（第一个需要分片的参数）
    base_len = None
    for a, ax in zip(args, chunk_axes):
        if ax is not None:
            base_len = a.shape[ax]
            break
    if base_len is None:
        # 没有任何参数需要分片，直接调用
        return cell(*args, **kwargs)

    # 检查其它分片参数的长度一致
    for a, ax in zip(args, chunk_axes):
        if ax is None:
            continue
        if a.shape[ax] != base_len:
            raise ValueError(f"Slice dimension length is not consistent: {a.shape[ax]} vs {base_len}")

    for k, v in kwargs.items():
        ax = kw_axes.get(k, None)
        if ax is not None:
            if not isinstance(v, Tensor):
                raise TypeError(f"kw {k} needs to be sliced but is not a Tensor")
            if v.shape[ax] != base_len:
                raise ValueError(f"kw {k} slice dimension length is not consistent: {v.shape[ax]} vs {base_len}")

    outputs = []
    n = int(base_len)
    for s in range(0, n, chunk_size):
        size = min(chunk_size, n - s)

        sliced_args = []
        for a, ax in zip(args, chunk_axes):
            if ax is None:
                sliced_args.append(a)
            else:
                sliced_args.append(_slice_along_axis(a, s, size, ax))

        sliced_kwargs = {}
        for k, v in kwargs.items():
            ax = kw_axes.get(k, None)
            if ax is None:
                sliced_kwargs[k] = v
            else:
                sliced_kwargs[k] = _slice_along_axis(v, s, size, ax)

        # 如需进一步省显存，可改成：out = ms.recompute(cell, *sliced_args, **sliced_kwargs)
        out = cell(*sliced_args, **sliced_kwargs)
        outputs.append(out)

    return _tree_concat(outputs, axis=out_axis)
