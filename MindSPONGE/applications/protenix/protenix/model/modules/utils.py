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

"""utils for the modules"""

from typing import Union, Optional
import math
import numpy as np
import mindspore as ms
from .graph import Aggregate


def broadcast_token_to_atom(
    x_token: ms.Tensor, atom_to_token_idx: ms.Tensor
) -> ms.Tensor:
    """Broadcast token-level embeddings to atom-level embeddings

    Args:
        x_token (ms.Tensor): token embedding
            [..., N_token, d]
        atom_to_token_idx (ms.Tensor): map atom idx to token idx
            [..., N_atom] or [N_atom]

    Returns:
        ms.Tensor: atom embedding
            [..., N_atom, d]
    """

    if len(atom_to_token_idx.shape) == 1:
        # shape = [N_atom], easy index
        return x_token[..., atom_to_token_idx, :]

    return batched_gather(
        data=x_token,
        inds=atom_to_token_idx,
        dim=-2,
        no_batch_dims=len(x_token.shape[:-2]),
    )


def broadcast_token_to_local_atom_pair(
    z_token: ms.Tensor,
    atom_to_token_idx: ms.Tensor,
    n_queries: int,
    n_keys: int,
    compute_mask: bool = True,
) -> ms.Tensor:
    """Broadcast token pair embedding to atom pair embedding

    Args:
        z_token (ms.Tensor): token pair embedding
            [..., N_token, N_token, d]
        atom_to_token_idx (ms.Tensor): map atom idx to token idx
            [N_atom]

    Returns:
        z_gathered_blocked (ms.Tensor): atom pair embedding, with local blocked shape
            [..., n_trunks, n_queries, n_keys, d]
        pad_mask (ms.Tensor):
            [n_trunks, n_queries, n_keys]
        q_pad_length (int)
    """

    # [N_atom] -> [n_trunks, n_queries] and [n_trunks, n_keys]
    atom_to_token_idx_q, atom_to_token_idx_k, pad_info = rearrange_qk_to_dense_trunk(
        atom_to_token_idx,
        atom_to_token_idx,
        dim_q=-1,
        dim_k=-1,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
    )

    z_gathered_blocked = gather_pair_embedding_in_dense_trunk(
        z_token, idx_q=atom_to_token_idx_q, idx_k=atom_to_token_idx_k
    )

    return z_gathered_blocked, pad_info


class AggregateAtomToToken(Aggregate):
    """Aggregate atom to token"""
    def __init__(self, mode='add'):
        super().__init__(mode=mode)

    def construct(self, atom_attr, idx, out=None, dim_size=None, mask=None):
        r"""
        Args:
            atom_attr (Tensor): The source tensor of node attributes.
            idx (Tensor): The indices of sample to scatter to.
            out (Tensor): The destination tensor. Default: None.
            dim_size (int): If `out` is not given, automatically create output with size `dim_size`. Default: None.
                out and dim_size cannot be both None.
            mask (Tensor): The mask of the node_attr tensor
        Returns:
            Tensor.
        """
        return self.scatter(atom_attr, idx, out=out, dim_size=dim_size, mask=mask)


def batched_gather(
    data: ms.Tensor, inds: ms.Tensor, dim: int = 0, no_batch_dims: int = 0
) -> ms.Tensor:
    """Gather data according to indices specify by inds

    Args:
        data (ms.Tensor): the input data
            [..., K, ...]
        inds (ms.Tensor): the indices for gathering data
            [..., N]
        dim (int, optional): along which dimension to gather data by inds (the dim of "K" "N"). Defaults to 0.
        no_batch_dims (int, optional): length of dimensions before the "dim" dimension. Defaults to 0.

    Returns:
        ms.Tensor: gathered data
            [..., N, ...]
    """

    # for the naive case
    if len(inds.shape) == 1 and no_batch_dims == 0 and dim == 0:
        return data[inds]

    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = ms.ops.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None)
                      for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def pad_at_dim(
    x: ms.Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> ms.Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (ms.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        ms.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return ms.ops.pad(x, padding=pad, value=value)


def move_final_dim_to_dim(x: ms.Tensor, dim: int) -> ms.Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (ms.Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        ms.Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)


def reshape_at_dim(
    x: ms.Tensor, dim: int, target_shape: Union[tuple[int], list[int]]
) -> ms.Tensor:
    """reshape dimension dim of x to target_shape

    Args:
        x (ms.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        ms.Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1:])
    return x.reshape(target_shape)


def unfold(x: ms.Tensor, dim: int, size: int, step: int) -> ms.Tensor:
    """
    Unfold a tensor along a specified dimension.

    Args:
        x (ms.Tensor): The input tensor.
        dim (int): The dimension along which to unfold.
        size (int): The size of the unfolding window.
        step (int): The step size for the unfolding.

    Returns:
        ms.Tensor: The unfolded tensor.
    """
    if x.ndim == 0:
        raise ValueError("unfold on a 0-D tensor is not defined.")
    if size <= 0 or step <= 0:
        raise ValueError("`size` and `step` must be positive integers.")

    orig_shape = x.shape
    ndim = x.ndim
    if dim < 0:
        dim += ndim
    if not 0 <= dim < ndim:
        raise ValueError(f"dim out of range: got {dim} for ndim={ndim}")

    length = orig_shape[dim]
    if length is None:
        raise ValueError(
            "This version requires known length along the unfold dim.")
    if size > length:
        raise ValueError(f"window size {size} > length {length} at dim {dim}")

    blocks = 1 + (length - size) // step
    if blocks <= 0:
        raise ValueError(
            "No valid window produced. Check `size/step` vs length.")

    pre = int(np.prod(orig_shape[:dim])) if dim > 0 else 1
    post = int(np.prod(orig_shape[dim+1:])) if dim+1 < ndim else 1
    x2 = ms.ops.reshape(x, (pre, length, post))
    starts = ms.ops.arange(0, blocks * step, step)      # [blocks]
    offs = ms.ops.arange(0, size, 1)                  # [size]
    idx = ms.ops.expand_dims(starts, 1) + \
        ms.ops.expand_dims(offs, 0)  # [blocks, size]
    idx = ms.ops.cast(idx, ms.int32)
    out = ms.ops.gather(x2, idx, axis=1)
    out = ms.ops.transpose(out, (0, 1, 3, 2))
    out_shape = tuple(orig_shape[:dim]) + (int(blocks),) + \
        tuple(orig_shape[dim+1:]) + (int(size),)
    out = ms.ops.reshape(out, out_shape)
    return out


def rearrange_qk_to_dense_trunk(
    q: Union[ms.Tensor, list[ms.Tensor]],
    k: Union[ms.Tensor, list[ms.Tensor]],
    dim_q: Union[int, list[int]],
    dim_k: Union[int, list[int]],
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
) -> tuple[Union[ms.Tensor, list[ms.Tensor]]]:
    """Rearrange q/k into blocked tensors for local operations.

    Args:
        q (ms.Tensor): query tensor. Could be a tensor or a list of tensors.
            [..., n_q, ...] (n_q is at dimension dim_q)
        k (ms.Tensor | List[ms.Tensor]): key tensor. Could be a tensor or a list of tensors.
            [..., n_k, ...] (n_k is at dimension dim_k)
        dim_q (int): along which dimension to build the trunks. Could be an int or a list of int.
        dim_k (int): along which dimension to build the trunks. Could be an int or a list of int.
        n_queries (int, optional): local window size of query tensor.
        n_keys (int, optional): local window size of key/value tensor.

    Returns:
        tuple[Union[ms.Tensor, list[ms.Tensor]]]:
            q_trunked: ms.Tensor or list of tensors. Same as the input type.
                [..., n_trunks, n_queries, ...]
            k_trunked: ms.Tensor or list of tensors. Same as the input type.
                [..., n_trunks, n_keys, ...]
            padding_info (dict):
                mask_trunked: ms.Tensor
                    [n_trunks, n_queries, n_keys]
                q_pad: query padded dimension
    """

    def basic_checks(x, dim_x):
        if isinstance(x, list):
            x_is_list = True
            if not isinstance(dim_x, list):
                raise TypeError("dim_x must be an int or a list of ints")
        else:
            x_is_list = False
            x = [x]
            dim_x = [dim_x]
        n_x = x[0].shape[dim_x[0]]
        for i, d in enumerate(dim_x):
            if d < 0:
                dim_x[i] = len(x[i].shape) + d
            if x[i].shape[dim_x[i]] != n_x:
                raise ValueError("x[i].shape[dim_x[i]] must be equal to n_x")
        return x, dim_x, x_is_list, n_x, len(x)

    q, dim_q, q_is_list, n, num_q = basic_checks(q, dim_q)
    k, dim_k, k_is_list, n_k, num_k = basic_checks(k, dim_k)

    if n != n_k:
        raise ValueError("n must be equal to n_k")
    n_trunks = int(math.ceil(n / n_queries))
    q_pad_length = n_trunks * n_queries - n

    q_new = [
        pad_at_dim(q[i], dim=dim_q[i], pad_length=(0, q_pad_length))
        for i in range(num_q)
    ]
    q_trunked = [
        reshape_at_dim(q_new[i], dim=dim_q[i],
                       target_shape=(n_trunks, n_queries))
        for i in range(num_q)
    ]

    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 1 / 2) * n_queries + n_keys / 2 - n + 1 / 2)

    k_new = [
        pad_at_dim(k[i], dim=dim_k[i], pad_length=(pad_left, pad_right))
        for i in range(num_k)
    ]
    k_trunked = [
        unfold(k_new[i], dim_k[i], size=n_keys, step=n_queries).astype(k_new[i].dtype) for i in range(num_k)
    ]
    k_trunked = [
        move_final_dim_to_dim(k_trunked[i], dim=dim_k[i] + 1) for i in range(num_k)
    ]

    if compute_mask:
        pad_mask = q[0].new_ones(
            (*(1,) * len(q[0].shape[:-2]),
             n + q_pad_length,
             n + pad_left + pad_right,)
        )
        pad_mask[..., :n, 0:pad_left] = 0
        pad_mask[..., :n, pad_left + n::] = 0
        pad_mask[..., n::, :] = 0

        concat_split_data = optimized_concat_split(pad_mask, n_queries)
        pad_mask_trunked = unfold(
            concat_split_data, -1, n_keys, pad_mask.shape[-1] + n_queries
        ).swapaxes(-2, -3).bool()
    else:
        pad_mask_trunked = None

    if not q_is_list:
        q_trunked = q_trunked[0]
    if not k_is_list:
        k_trunked = k_trunked[0]

    padding_info = {
        "mask_trunked": pad_mask_trunked,
        "q_pad": q_pad_length,
        "k_pad_left": pad_left,
        "k_pad_right": pad_right,
    }

    return q_trunked, k_trunked, padding_info


def rearrange_to_dense_trunk(
    q: ms.Tensor,
    k: ms.Tensor,
    v: ms.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[ms.Tensor] = None,
    inf: float = 1e10,
) -> tuple[Union[ms.Tensor, int]]:
    """Rearrange q/k/v/bias into blocked tensors for local attention.

    Args:
        q (ms.Tensor): query tensor
            [..., n_q, d]
        k (ms.Tensor): key tensor
            [..., n_kv, d]
        v (ms.Tensor): value tensor
            [..., n_kv, d]
        attn_bias (ms.Tensor, optional): attention bias
            [..., n_q, n_kv] or None
        n_queries (int, optional): local window size of query tensor.
        n_keys (int, optional): local window size of key/value tensor.
        inf (float, optional): used for attention masking. Defaults to 1e10.

    Returns:
        tuple[Union[ms.Tensor, int]]:
            q_trunked
                [..., n_trunks, n_queries, d]
            k_trunked / v_trunked
                [..., n_trunks, n_keys, d]
            attn_bias_trunked:  padded position filled with -inf
                [..., n_trunks, n_queries, n_keys]
            q_pad_length: query padded dimension
    """

    n, _ = q.shape[-2:]

    q_trunked, kv_trunked, padding_info = rearrange_qk_to_dense_trunk(
        q=q,
        k=[k, v],
        dim_q=-2,
        dim_k=[-2, -2],
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=False,
    )
    q_pad_length, pad_left, pad_right = (
        padding_info["q_pad"],
        padding_info["k_pad_left"],
        padding_info["k_pad_right"],
    )

    # Padded_width = n + pad_left + pad_right
    if attn_bias is None:
        attn_bias = q.new_zeros(
            (*(1,) * len(q.shape[:-2]), n +
             q_pad_length, n + pad_left + pad_right)
        )
        attn_bias[..., :n, 0:pad_left] = -inf
        attn_bias[..., :n, pad_left + n::] = -inf
        attn_bias[..., n::, :] = -inf
    else:
        attn_bias = ms.ops.pad(
            attn_bias, (pad_left, pad_right, 0, q_pad_length), value=-inf)

    concat_split_data = optimized_concat_split(attn_bias, n_queries)
    attn_bias_trunked = unfold(
        concat_split_data, -1, n_keys, attn_bias.shape[-1] + n_queries
    ).transpose(-2, -3)
    return q_trunked, kv_trunked[0], kv_trunked[1], attn_bias_trunked, q_pad_length


def optimized_concat_split(attn_bias: ms.Tensor, n_queries: int) -> ms.Tensor:
    """Optimized concatenation and splitting of attention bias tensor.

    Args:
        attn_bias (ms.Tensor): The attention bias tensor.
            Shape: [..., d, e]
        n_queries (int): The number of queries in each split.

    Returns:
        ms.Tensor: The reshaped and permuted attention bias tensor.
            Shape: [..., n_queries, d // n_queries * e]
    """
    d = attn_bias.shape[-2]
    e = attn_bias.shape[-1]
    if d % n_queries != 0:
        raise ValueError("d must be divisible by n_queries")
    num_splits = d // n_queries
    reshaped = attn_bias.reshape(
        *attn_bias.shape[:-2], num_splits, n_queries, e)
    permuted = reshaped.permute(*range(reshaped.dim() - 3), -2, -3, -1)
    output = permuted.reshape(*attn_bias.shape[:-2], n_queries, num_splits * e)
    return output


def gather_pair_embedding_in_dense_trunk(
    x: ms.Tensor, idx_q: ms.Tensor, idx_k: ms.Tensor
):
    """
    Selectively gather elements from a tensor using two sets of indices.

        x: [..., N_token, N_token, d]
        idx_q: [N_b, N_q]
        idx_k: [N_b, N_k]

    Return:
        y: [..., N_b, N_q, N_k, d]
            where y[..., b, i, j, :] = x[..., idx_q[b, i], idx_k[b, j], :]
    """
    idx_q = idx_q.astype(ms.int32)
    idx_k = idx_k.astype(ms.int32)
    if not len(idx_q.shape) == len(idx_k.shape) == 2:
        raise ValueError("idx_q and idx_k must both be 2-dimensional tensors")

    # Get the shape parameters
    _, n_q = idx_q.shape
    n_k = idx_k.shape[1]

    # Expand idx_q and idx_k to match the shape required for advanced indexing
    idx_q_expanded = idx_q.unsqueeze(-1).expand((-1, -1, n_k))
    idx_k_expanded = idx_k.unsqueeze(1).expand((-1, n_q, -1))

    # Use advanced indexing to gather the desired elements
    y = x[..., idx_q_expanded, idx_k_expanded, :]

    return y
