# Copyright 2024 Huawei Technologies Co., Ltd
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
"""gemnet preprocess"""
import numpy as np
import mindspore as ms
from .data_utils import (
    get_pbc_distances, radius_graph_pbc, frac_to_cart_coords_numpy)
from .utils import real_sph_harm_np


class GemNetPreprocess:
    """
    Preprocess for decoder
    Args:
        otf_graph (bool): Whether build otf_graph. Default: False.
        cutoff (float): Cutoff length for edges. Default: 7.0
        max_num_neighbors (int): Max number of neighbors for an atom. Default: 20.
    """

    def __init__(self,
                 otf_graph=False,
                 cutoff=7.0,
                 max_num_neighbors=20,):
        self.cutoff = cutoff
        self.max_neighbors = max_num_neighbors
        self.otf_graph = otf_graph

    @staticmethod
    def get_triplets(edge_index):
        r"""
        compute triplets from edge_index

        Args:
            edge_index (numpy.ndarray): The shape of tensor is :math:`(2, total\_edges)`.

        Returns:
            idx_kj (numpy.ndarray): First edge of triplets.
            The shape of tensor is :math:`(total\_triplets,)`.
            idx_ji (numpy.ndarray): Second edge of triplets.
            The shape of tensor is :math:`(total\_triplets,)`.
            id3_ragged_index (numpy.ndarray): The shape of tensor is :math:`(total\_triplets,)`.
        """
        row, col = edge_index
        int1d = np.intersect1d(row, col)
        idx_kj = [0] * int1d.size
        idx_ji = [0] * int1d.size

        for pos in range(len(int1d)):
            kj = np.where(int1d[pos] == col)[0]
            ji = np.where(int1d[pos] == row)[0]
            idx_kj[pos] = kj.repeat(ji.size)
            idx_ji[pos] = np.tile(ji[::-1], kj.size)
        idx_kj = np.concatenate(idx_kj)
        idx_ji = np.concatenate(idx_ji)
        mask = idx_ji != idx_kj
        idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]
        # if needed: idx_i, idx_j, idx_k can get using col[idx_ji], col[idx_kj], row[idx_kj]
        num_triplets = np.bincount(idx_ji, minlength=col.shape[0])
        id3_ragged_idx = ragged_range(num_triplets)
        return idx_kj, idx_ji, id3_ragged_idx

    @staticmethod
    def select_symmetric_edges(edge_attr, mask, reorder_idx, inverse_neg):
        r"""
        select_symmetric_edges.

        Args:
            edge_attr (numpy.ndarray): The shape of array is :math:`(total\_edges, 3)`.
            mask (numpy.ndarray): The shape of array is :math:`(total\_edges,)`.
            reorder_idx (numpy.ndarray): The shape of array is :math:`(total\_edges,)`.
            inverse_neg (bool): Whether inverse negative.

        Returns:
            edge_attr_ordered (numpy.ndarray): Reordered edge attribution.
            The shape of array is :math:`(total\_edges, 3)`.
        """
        # Mask out counter-edges
        edge_attr_directed = edge_attr[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        edge_attr_cat = np.concatenate(
            [edge_attr_directed, sign * edge_attr_directed])
        # Reorder everything so the edges of every image are consecutive
        edge_attr_ordered = edge_attr_cat[reorder_idx.astype(np.int32)]
        return edge_attr_ordered

    def generate_interaction_graph(self, cart_coords, lengths, angles,
                                   num_atoms, edge_index, to_jimages,
                                   num_bonds):
        r"""
        generate interaction graph

        Args:
            cart_coords (numpy.ndarray): The shape of array is :math:`(total\_atoms, 3)`.
            lengths (numpy.ndarray): The shape of array is :math:`(batch\_size, 3)`.
            angles (numpy.ndarray): The shape of array is :math:`(batch\_size, 3)`.
            num_atoms (numpy.ndarray): The shape of array is :math:`(batch\_size)`.
            edge_index (numpy.ndarray): The shape of array is :math:`(2, total\_edges)`.
            to_jimages (numpy.ndarray): The shape of array is :math:`(total\_edges)`.
            num_bonds (numpy.ndarray): The shape of array is :math:`(batch\_size)`.

        Return: (tuple)
            edge_index (numpy.ndarray): The shape of array is :math:`(2, total\_edges)`.
            neighbors (numpy.ndarray): The shape of array is :math:`(batch\_size,)`.
            d_st (numpy.ndarray): The shape of array is :math:`(total\_edges,)`.
            v_st (numpy.ndarray): The shape of array is :math:`(total\_edges, 3)`.
            id_swap (numpy.ndarray): The shape of array is :math:`(total\_edges,)`.
            id3_ba (numpy.ndarray): The shape of array is :math:`(total\_triplets,)`.
            id3_ca (numpy.ndarray): The shape of array is :math:`(total\_triplets,)`.
            id3_ragged_idx (numpy.ndarray): The shape of array is :math:`(total\_triplets,)`.
        """

        if self.otf_graph:
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords, lengths, angles, num_atoms, self.cutoff, self.max_neighbors)

        # Switch the indices, so the second one becomes the target index,
        # over which we can efficiently aggregate.
        out = get_pbc_distances(
            cart_coords,
            edge_index,
            lengths,
            angles,
            to_jimages,
            num_atoms,
            num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        edge_index = out["edge_index"]
        d_st = out["distances"]
        # These vectors actually point in the opposite direction.
        # But we want to use col as idx_t for efficient aggregation.
        v_st = -out["distance_vec"] / d_st[:, None]

        (
            edge_index,
            _,
            neighbors,
            d_st,
            v_st,
        ) = self.reorder_symmetric_edges(
            edge_index, to_jimages, num_bonds, d_st, v_st
        )
        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )
        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(edge_index)
        return (
            edge_index,
            neighbors,
            d_st,
            v_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

    def graph_generation(self, frac_coords, num_atoms, lengths, angles,
                         edge_index, to_jimages, num_bonds):
        r"""
        graph generation

        Args:
            frac_coords (numpy.ndarray): The shape of array is :math:`(total\_atoms, 3)`.
            num_atoms (numpy.ndarray): The shape of array is :math:`(batch\_size,)`.
            lengths (numpy.ndarray): The shape of array is :math:`(batch\_size, 3)`.
            angles (numpy.ndarray): The shape of array is :math:`(batch\_size, 3)`.
            edge_index (numpy.ndarray): The shape of array is :math:`(2, total\_edges)`.
            to_jimages (numpy.ndarray): The shape of array is :math:`(total\_edges)`.
            num_bonds (numpy.ndarray): The shape of array is :math:`(batch\_size,)`.

        Return: (tuple)
            edge_index (Tensor): The shape of array is :math:`(2, total\_edges)`.
            idx_s (Tensor): The shape of array is :math:`(total\_edges,)`.
            idx_t (Tensor): The shape of array is :math:`(total\_edges,)`.
            id3_ba (Tensor): The shape of array is :math:`(total\_triplets)`.
            id3_ca (Tensor): The shape of array is :math:`(total\_triplets)`.
            id3_ragged_idx (Tensor): The shape of array is :math:`(total\_triplets)`.
            id3_ragged_idx_max (int): Maximum value of id3_ragged_idx.
            cosfi_cab (Tensor): The shape of array is :math:`(total\_triplets)`.
            d_st (Tensor): The shape of array is :math:`(total\_edges,)`.
            v_st (Tensor): The shape of array is :math:`(total\_edges, 3)`.
            id_swap (Tensor): The shape of array is :math:`(total\_edges,)`.
            y_l_m (Tensor): The shape of array is :math:`(total\_triplets)`.
        """
        pos = frac_to_cart_coords_numpy(
            frac_coords, lengths, angles, num_atoms)
        (
            edge_index,
            _,
            d_st,
            v_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(
            pos, lengths, angles, num_atoms, edge_index, to_jimages,
            num_bonds)
        idx_s, idx_t = edge_index
        # Calculate triplet angles
        cosfi_cab = inner_product_normalized(v_st[id3_ca], v_st[id3_ba])
        # numpy to mindspore Tensor
        id3_ragged_idx_max = id3_ragged_idx.max().astype(np.int32).item()
        y_l_m = real_sph_harm_np(
            cosfi_cab, 7, use_theta=False, zero_m_only=True)
        y_l_m = ms.Tensor(y_l_m, ms.float32)
        idx_s = ms.Tensor(idx_s, ms.int32)
        idx_t = ms.Tensor(idx_t, ms.int32)
        id3_ca = ms.Tensor(id3_ca, ms.int32)
        id3_ba = ms.Tensor(id3_ba, ms.int32)
        id3_ragged_idx = ms.Tensor(id3_ragged_idx, ms.int32)
        d_st = ms.Tensor(d_st, ms.float32)
        v_st = ms.Tensor(v_st, ms.float32)
        id_swap = ms.Tensor(id_swap, ms.int32)
        return (edge_index, idx_s, idx_t, id3_ca, id3_ba,
                id3_ragged_idx, id3_ragged_idx_max,
                cosfi_cab, d_st, v_st, id_swap, y_l_m)

    def reorder_symmetric_edges(
            self, edge_index, cell_offsets, neighbors, edge_dist, edge_vector
    ):
        r"""
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.

        Args:
            edge_index (numpy.ndarray): The shape of array is :math:`(2, total\_edges)`.
            cell_offsets (numpy.ndarray): The shape of array is :math:`(total\_edges, 3)`.
            neighbors (numpy.ndarray): The shape of array is :math:`(batch\_size,)`.
            edge_dist (numpy.ndarray): The shape of array is :math:`(total\_edges,)`.
            edge_vector (numpy.ndarray): The shape of array is :math:`(total\_edges, 3)`.

        Returns:
            edge_index_new (numpy.ndarray): The shape of array is :math:`(2, total\_edges)`.
            cell_offsets_new (numpy.ndarray): The shape of array is :math:`(total\_edges, 3)`.
            neighbors_new (numpy.ndarray): The shape of array is :math:`(batch\_size,)`.
            edge_dist_new (numpy.ndarray): The shape of array is :math:`(total\_edges,)`.
            edge_vector_new (numpy.ndarray): The shape of array is :math:`(total\_edges, 3)`.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[np.broadcast_to(
            mask[None, :], (2, mask[None, :].shape[1]))].reshape(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = np.concatenate(
            (
                edge_index_new,
                np.stack([edge_index_new[1], edge_index_new[0]], axis=0),
            ),
            axis=1,
        )

        # Count remaining edges per image
        batch_edge = np.repeat(
            np.arange(neighbors.shape[0]),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * np.bincount(
            batch_edge, minlength=neighbors.shape[0]
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.shape[1],
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx.astype(np.int32)]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )


def ragged_range(sizes):
    r"""Multiple concatenated ranges.

    Args:
        sizes (numpy.ndarray): The shape of array is :math:`(n,)`.

    Returns:
        numpy.ndarray: The shape of array is :math:`(\sum(sizes),)`.

    Examples:
        >>> sizes = np.array([1 4 2 3])
        >>> out = ragged_range(sizes)
        >>> print(out)
        [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.ndim == 1

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not np.all(sizes_nonzero):
        sizes = sizes[sizes_nonzero]

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    id_steps = np.ones(sizes.sum().astype(np.int32), dtype=np.float32)
    id_steps[0] = 0
    insert_index = np.cumsum(sizes[:-1], axis=0)
    insert_val = (1 - sizes)[:-1]

    # Assign index-offsetting values
    id_steps[insert_index.astype(np.int32)] = insert_val

    # Finally index into input array for the group repeated o/p
    res = np.cumsum(id_steps, axis=0)
    return res


def segment_csr(src, index, reduce="sum"):
    """segment_csr"""
    out_shape = src.shape
    out_shape[0] = len(index) - 1
    out = np.zeros(out_shape)
    count = index[1:] - index[:-1]
    n = 1
    for i in range(src.shape[0]):
        if i < index[n]:
            out[index[n - 1]] += src[i]
        else:
            out[index[n]] += src[i]
            n += 1
    if reduce == "mean":
        out = out.T.div(count).T
    return out


def repeat_blocks(
        sizes,
        repeats,
        continuous_indexing=True,
        start_idx=0,
        block_inc=0,
        repeat_inc=0,
):
    """
    Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    Args:
        sizes (numpy.ndarray): Number of elements in each block.
        repeats (numpy.ndarray): Number of times to repeat each block.
        continuous_indexing (bool): Whether to keep increasing the index after each block
        start_idx (int): Starting index
        block_inc (int or numpy.ndarray): Number to increment by after each block,
            either global or per block.
        repeat_inc (int or numpy.ndarray): Number to increment by after each repetition,
            either global or per block.

    Returns:
        numpy.ndarray: The shape of array is :math:`(sum(sizes * repeats),)`.

    Examples:
        >>> sizes = np.array([1, 3, 2])
        >>> repeats = np.array([3, 2, 3])
        >>> out = repeat_blocks(sizes, repeats, continuous_indexing = False)
        >>> print(out)
        [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        >>> out = repeat_blocks(sizes, repeats, continuous_indexing = True)
        >>> print(out)
        [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        >>> out = repeat_blocks(sizes, repeats, continuous_indexing = True, repeat_inc = 4)
        >>> print(out)
        [0 0 0  1 2 3 1 2 3  4 5 8 9 12 13]
        >>> out = repeat_blocks(sizes, repeats, continuous_indexing = True, start_idx = 5)
        >>> print(out)
        [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        >>> out = repeat_blocks(sizes, repeats, continuous_indexing = True, block_inc = 1)
        >>> print(out)
        [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        >>> sizes = np.array([0, 3, 2])
        >>> repeats = np.array([3, 2, 3])
        >>> out = repeat_blocks(sizes, repeats, continuous_indexing = True)
        >>> print(out)
        [0 1 2 0 1 2  3 4 3 4 3 4]
        >>> sizes = np.array([2, 3, 2])
        >>> repeats = np.array([2, 0, 2])
        >>> out = repeat_blocks(sizes, repeats, continuous_indexing = True)
        >>> print(out)
        [0 1 0 1  5 6 5 6]
    """
    assert sizes.ndim == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not np.all(sizes_nonzero):
        assert block_inc == 0
        sizes = sizes[sizes_nonzero]
        if isinstance(repeats, np.ndarray):
            repeats = repeats[sizes_nonzero]
        if isinstance(repeat_inc, np.ndarray):
            repeat_inc = repeat_inc[sizes_nonzero]

    if isinstance(repeats, np.ndarray):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = np.concatenate((one, sizes))
            repeats = np.concatenate((one, repeats))
            if isinstance(block_inc, np.ndarray):
                block_inc = np.concatenate((zero, block_inc))
            if isinstance(repeat_inc, np.ndarray):
                repeat_inc = np.concatenate((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = np.repeat(
        np.arange(len(sizes)), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    total_size = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = np.ones(total_size.astype(np.int64))
    id_ar[0] = 0
    insert_index = np.cumsum(sizes[r1[:-1]], axis=0).astype(np.int32)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, np.ndarray) and np.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = np.concatenate((sizes.new_zeros(1), np.cumsum(diffs, axis=0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, np.ndarray):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group"s
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, np.ndarray):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, np.ndarray):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, np.ndarray):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val_mask = r1[1:] != r1[:-1]
    insert_val[insert_val_mask] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = np.cumsum(id_ar, axis=0)
    return res


def inner_product_normalized(x, y):
    """
    Calculate the inner product between the given normalized vectors,
    giving a result between -1 and 1.
    """
    return np.clip(np.sum(x * y, axis=-1), a_min=-1, a_max=1)
