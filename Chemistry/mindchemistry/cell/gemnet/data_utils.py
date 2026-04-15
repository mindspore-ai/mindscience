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
"""data_utils"""
import copy
import numpy as np
import mindspore as ms
import mindspore.mint as mint


# Tensor of unit cells. Assumes 27 cells in -1, 0, 1 offsets in the x and y dimensions
# Note that differing from OCP, we have 27 offsets here because we are in 3D
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.

    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.

    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    r"""
    Converts lattice from length and angles to matrix.

    Args:
        a (numpy.ndarray): a lattice parameter.
        b (numpy.ndarray): b lattice parameter.
        c (numpy.ndarray): c lattice parameter.
        alpha (numpy.ndarray): alpha lattice angle.
        beta (numpy.ndarray): beta lattice angle.
        gamma (numpy.ndarray): gamma lattice angle.

    returns:
        (numpy.ndarray): Lattice vector of each samples.
        The shape of Tensor is :math:`(3, batch\_size)`.
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, _ = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_params_to_matrix_numpy(lengths, angles):
    """
    Compute lattice matrix from params.

    Args:
        lengths (numpy.ndarray): shape (N, 3), unit A
        angles (numpy.ndarray): shape (N, 3), unit degree

    Returns:
        (numpy.ndarray): shape (N, 3, 3), unit A
    """
    angles_r = np.deg2rad(angles).reshape((-1, 3))
    coses = np.cos(angles_r)
    sins = np.sin(angles_r)
    lengths = lengths.reshape((-1, 3))

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = np.clip(val, -1., 1.)
    gamma_star = np.arccos(val)

    vector_a = np.stack([
        lengths[:, 0] * sins[:, 1],
        np.zeros(lengths.shape[0]),
        lengths[:, 0] * coses[:, 1]], axis=1)
    vector_b = np.stack([
        -lengths[:, 1] * sins[:, 0] * np.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * np.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], axis=1)
    vector_c = np.stack([
        np.zeros(lengths.shape[0]),
        np.zeros(lengths.shape[0]),
        lengths[:, 2]], axis=1)

    return np.stack([vector_a, vector_b, vector_c], axis=1)


def frac_to_cart_coords_numpy(
        frac_coords,
        lengths,
        angles,
        num_atoms,
):
    """
    Turned the pos from frac coords to cart coords

    Args:
        frac_coords (numpy.ndarray): frac position of each atom
        length (numpy.ndarray): a, b, c for each cystal
        angles (numpy.ndarray): alpha, beta, gamma for each cystal
        num_atoms (numpy.ndarray): number of atoms for each cystal

    Returns:
        pos (numpy.ndarray): The cart coords of each atom.
    """

    lattice = lattice_params_to_matrix_numpy(lengths, angles)
    lattice_nodes = np.repeat(lattice, num_atoms, axis=0)
    pos = np.einsum("bi,bij->bj", frac_coords, lattice_nodes)

    return pos


def cart_to_frac_coords_numpy(
        cart_coords,
        lengths,
        angles,
        num_atoms,
):
    """
    Turned the pos from cart coords to frac coords

    Args:
        cart_coords (numpy.ndarray): cart coords of each atom
        lengths (numpy.ndarray): a, b, c for each cystal
        angles (numpy.ndarray): alpha, beta, gamma for each cystal
        num_atoms (numpy.ndarray): number of atoms for each cystal

    return:
        frac_coords (numpy.ndarray): frac coords of each atom.
    """

    lattice = lattice_params_to_matrix_numpy(lengths, angles)
    inv_lattice = np.linalg.pinv(lattice)
    inv_lattice_nodes = np.repeat(inv_lattice, num_atoms, axis=0)
    frac_coords = np.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    return frac_coords % 1.


def get_pbc_distances(
        coords,
        edge_index,
        lengths,
        angles,
        to_jimages,
        num_atoms,
        num_bonds,
        coord_is_cart=False,
        return_offsets=False,
        return_distance_vec=False,
):
    """
    get pbc distances

    Args:
        coords (numpy.ndarray): position of each atom
        edge_index (numpy.ndarray): edge index
        lengths (numpy.ndarray): lattice constant for each cystal
        angles (numpy.ndarray): alpha, beta, gamma for each cystal
        to_jimages (numpy.ndarray): Based on peroiodic boundary conditions
        num_atoms (numpy.ndarray): number of atoms for each cystal
        num_bonds (numpy.ndarray): number of bonds for each cystal
        coord_is_cart (bool): whether the coords are in cartesian coordinates
        return_distance_vec (bool): whether to return the distance vector
        return_offsets (bool): whether to return the offsets

    Returns:(dict):
        edge_index (numpy.ndarray): Edge index.
        distances (numpy.ndarray): Dinstance between nodes.
        distance_vec (numpy.ndarray, optional): Distance vector between nodes.
        offsets (numpy.ndarray, optional): Offsets of nodes.
    """
    lattice = lattice_params_to_matrix_numpy(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = np.repeat(lattice, num_atoms, axis=0)
        pos = np.einsum("bi,bij->bj", coords, lattice_nodes)

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = np.repeat(lattice, num_bonds, axis=0)
    offsets = np.einsum("bi,bij->bj", to_jimages, lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = np.linalg.norm(distance_vectors, axis=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


def radius_graph_pbc(cart_coords, lengths, angles, num_atoms,
                     radius, max_num_neighbors_threshold):
    """
    Computes pbc graph edges under pbc.

    Args:
        cart_coords (numpy.ndarray): position of each atom
        lengths (numpy.ndarray): lattice constant for each cystal
        angles (numpy.ndarray): alpha, beta, gamma for each cystal
        num_atoms (numpy.ndarray): number of atoms for each cystal
        radius (float): radius
        max_num_neighbors_threshold (int): max number of neighbors

    Returns:
        (numpy.ndarray): edge_index.
        (numpy.ndarray): unit_cell of each sample.
        (numpy.ndarray): number of neighbors of each atoms.
    """
    batch_size = len(num_atoms)

    # position of the atoms
    atom_pos = cart_coords

    # Before computing the pairwise distances between atoms,
    # first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2)

    # index offset between images
    index_offset = (
        np.cumsum(num_atoms_per_image, axis=0) - num_atoms_per_image
    )

    index_offset_expand = np.repeat(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = np.repeat(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    num_atom_pairs = np.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        np.cumsum(num_atoms_per_image_sqr, axis=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = np.repeat(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        np.arange(num_atom_pairs) - index_sqr_offset
    )

    index1, index2, mask, atom_distance_sqr, unit_cell_per_atom = get_pair_index(
        atom_count_sqr, num_atoms_per_image_expand, index_offset_expand, atom_pos,
        batch_size, lengths, angles, radius, num_atoms_per_image_sqr)

    unit_cell = unit_cell_per_atom.reshape(-1, 3)[np.broadcast_to(
        mask.reshape(-1, 1), (mask.reshape(-1, 1).shape[0], 3))]

    unit_cell = unit_cell.reshape(-1, 3)

    num_neighbors = np.bincount(index1)
    max_num_neighbors = np.max(num_neighbors)

    # Compute neighbors per image
    new_max_neighbors = copy.deepcopy(num_neighbors)
    new_max_neighbors[
        new_max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    new_num_neighbors = np.zeros(len(cart_coords) + 1).astype(np.int32)
    new_natoms = np.zeros(num_atoms.shape[0] + 1).astype(np.int32)
    new_num_neighbors[1:] = np.cumsum(new_max_neighbors, axis=0)
    new_natoms[1:] = np.cumsum(num_atoms, axis=0)
    num_neighbors_image = (
        new_num_neighbors[new_natoms[1:]] - new_num_neighbors[new_natoms[:-1]]
    )

    # If max_num_neighbors is below the threshold, return early
    if (max_num_neighbors <= max_num_neighbors_threshold
            or max_num_neighbors_threshold <= 0):
        return np.stack((index2, index1)), unit_cell, num_neighbors_image

    edge_index, unit_cell = edge_select_pbc(atom_distance_sqr, mask, cart_coords, max_num_neighbors,
                                            radius, num_neighbors, max_num_neighbors_threshold,
                                            index1, index2, unit_cell)

    return edge_index, unit_cell, num_neighbors_image


def get_pair_index(atom_count_sqr, num_atoms_per_image_expand, index_offset_expand, atom_pos,
                   batch_size, lengths, angles, radius, num_atoms_per_image_sqr):
    """
    Compute the indices for the pairs of atoms (using division and mod)
    If the systems get too large this apporach could run into numerical precision issues.

    Args:
        atom_count_sqr (numpy.ndarray): The squared distance between atoms.
        num_atoms_per_image_expand (numpy.ndarray): The number of atoms per image.
        index_offset_expand (numpy.ndarray): The index offset.
        atom_pos (numpy.ndarray): The position of each atom.
        batch_size (int): The batch size.
        lengths (numpy.ndarray): The lengths of the cell.
        angles (numpy.ndarray): The angles of the cell.
        radius (float): The radius.
        num_atoms_per_image_sqr (numpy.ndarray): The number of atoms per image squared.

    Returns:
        (numpy.ndarray): The index of the first atom.
        (numpy.ndarray): The index of the second atom.
        (numpy.ndarray): The mask.
        (numpy.ndarray): The squared distance between atoms.
        (numpy.ndarray): The unit cell.
    """
    index1 = (
        (atom_count_sqr // num_atoms_per_image_expand)
    ).astype(np.int32) + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ).astype(np.int32) + index_offset_expand
    # Get the positions for each atom
    pos1 = atom_pos[index1]
    pos2 = atom_pos[index2]

    unit_cell = np.array(OFFSET_LIST).astype(float)
    num_cells = len(unit_cell)
    unit_cell_per_atom = np.tile(unit_cell.reshape(
        (1, num_cells, 3)), (len(index2), 1, 1))
    unit_cell = np.swapaxes(unit_cell, 0, 1)
    unit_cell_batch = np.broadcast_to(unit_cell.reshape(1, 3, num_cells),
                                      (batch_size, unit_cell.shape[0], unit_cell.shape[1]))

    # lattice matrix
    lattice = lattice_params_to_matrix_numpy(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = np.swapaxes(lattice, 1, 2)
    pbc_offsets = np.matmul(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = np.repeat(
        pbc_offsets, num_atoms_per_image_sqr, axis=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = np.broadcast_to(pos1.reshape(-1, 3, 1),
                           (pos1.shape[0], pos1.shape[1], num_cells))
    pos2 = np.broadcast_to(pos2.reshape(-1, 3, 1),
                           (pos2.shape[0], pos2.shape[1], num_cells))
    index1 = np.tile(index1.reshape(-1, 1), (1, num_cells)).reshape(-1)
    index2 = np.tile(index2.reshape(-1, 1), (1, num_cells)).reshape(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = np.sum((pos1 - pos2) ** 2, axis=1)

    atom_distance_sqr = atom_distance_sqr.reshape(-1)

    # Remove pairs that are too far apart
    mask_within_radius = atom_distance_sqr <= radius * radius
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = atom_distance_sqr >= 0.0001
    mask = np.logical_and(mask_within_radius, mask_not_same)
    index1_new = index1[mask]
    index2_new = index2[mask]
    # if there is an atom with no neighbors, leave the nearest neighbor
    num_neighbors = np.bincount(index1_new)
    while 0 in num_neighbors:
        idx_0 = np.where(num_neighbors == 0)[0][0]
        idx_0_unmask = np.where(index1 == idx_0)[0][0]
        mask[idx_0_unmask] = True
        atom_distance_sqr[idx_0_unmask] = radius * radius - 0.1
        index1_new = index1[mask]
        index2_new = index2[mask]
        num_neighbors = np.bincount(index1_new)
    index1 = index1_new
    index2 = index2_new
    return index1, index2, mask, atom_distance_sqr, unit_cell_per_atom


def edge_select_pbc(atom_distance_sqr, mask, cart_coords, max_num_neighbors, radius, num_neighbors,
                    max_num_neighbors_threshold, index1, index2, unit_cell):
    """
    If the number of neighbors is greater than the threshold, select the nearest neighbors.

    Args:
        atom_distance_sqr (numpy.ndarray): The squared distance between atoms.
        mask (numpy.ndarray): The mask of the atoms.
        cart_coords (numpy.ndarray): The position of each atom.
        max_num_neighbors (int): The max number of neighbors.
        radius (float): The radius.
        num_neighbors (numpy.ndarray): The number of neighbors.
        max_num_neighbors_threshold (int): The max number of neighbors threshold.

    Returns:
        (numpy.ndarray): The edge index.
        (numpy.ndarray): The unit cell.
    """
    atom_distance_sqr = atom_distance_sqr[mask]

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.

    distance_sort = np.zeros(len(cart_coords) * max_num_neighbors)
    distance_sort.fill(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = np.cumsum(num_neighbors, axis=0) - num_neighbors
    index_neighbor_offset_expand = np.repeat(
        index_neighbor_offset, num_neighbors)
    index_sort_map = (
        index1 * max_num_neighbors
        + np.arange(len(index1))
        - index_neighbor_offset_expand
    )
    distance_sort[index_sort_map] = atom_distance_sqr
    distance_sort = distance_sort.reshape(len(cart_coords), max_num_neighbors)

    # Sort neighboring atoms based on distance
    index_sort = np.argsort(distance_sort, axis=1)
    distance_sort = np.sort(distance_sort, axis=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + np.broadcast_to(index_neighbor_offset.reshape(-1, 1),
                                              (index_neighbor_offset.reshape(-1, 1).shape[0],
                                               max_num_neighbors_threshold))
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = distance_sort <= radius * radius
    index_sort = index_sort[mask_within_radius]

    # At this point index_sort contains the index into index1 of
    # the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = np.zeros(len(index1)).astype(bool)
    mask_num_neighbors[index_sort] = True

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = index1[mask_num_neighbors]
    index2 = index2[mask_num_neighbors]
    unit_cell_index = np.broadcast_to(
        mask_num_neighbors.reshape(-1, 1), (mask_num_neighbors.reshape(-1, 1).shape[0], 3))
    unit_cell = unit_cell.reshape(-1, 3)[unit_cell_index]
    unit_cell = unit_cell.reshape(-1, 3)

    edge_index = np.stack((index2, index1))
    return edge_index, unit_cell


def lattice_params_to_matrix_mindspore(lengths, angles):
    """
    MindSpore version to compute lattice matrix from params.

    Args:
        lengths (Tensor): unit A. The shape of tensor is :math:`(N, 3)`.
        angles (Tensor): unit degree. The shape of tensor is :math:`(N, 3)`.

    Returns:
        (Tensor): unit A. The shape of tensor is :math:`(N, 3)`.
    """
    angles_r = ms.ops.deg2rad(angles)
    coses = mint.cos(angles_r)
    sins = mint.sin(angles_r)

    coses_0, coses_1, coses_2 = mint.split(coses, 1, 1)
    sins_0, sins_1, _ = mint.split(sins, 1, 1)
    lengths_0, lengths_1, lengths_2 = mint.split(lengths, 1, 1)
    val = mint.div((mint.mul(coses_0, coses_1) - coses_2),
                   mint.mul(sins_0, sins_1))
    val = mint.clamp(val, -0.9999, 0.9999)
    gamma_star = ms.ops.acos(val)

    vector_a = mint.stack([
        lengths_0 * sins_1,
        mint.zeros((lengths.shape[0], 1)),
        lengths_0 * coses_1], dim=1).squeeze()
    vector_b = mint.stack([
        -lengths_1 * sins_0 * mint.cos(gamma_star),
        lengths_1 * sins_0 * mint.sin(gamma_star),
        lengths_1 * coses_0], dim=1).squeeze()
    vector_c = mint.stack([
        mint.zeros((lengths.shape[0], 1)),
        mint.zeros((lengths.shape[0], 1)),
        lengths_2], dim=1).squeeze()

    return mint.stack([vector_a, vector_b, vector_c], dim=1)


def frac_to_cart_coords(
        frac_coords,
        lengths,
        angles,
        batch,
        lift_globaltonode
):
    r"""
    Turned the pos from fraction coords to cart coords

    Args:
        frac_coords (Tensor): frac position of each atom for each crystal.
            The shape of tensor is :math:`(total\_atoms, 3)`.
        lengths (Tensor): a, b, c for each cystal.
            The shape of tensor is :math:`(batch\_size, 3)`.
        angles (Tensor): alpha, beta, gamma for each cystal
            The shape of tensor is :math:`(batch\_size, 3)`.
        num_atoms (Tensor): number of atoms for each cystal
            The shape of tensor is :math:`(batch\_size)`.

    return:
        frac_coords (Tensor): cart coords of each atom for each crystal.
        The shape of tensor is :math:`(total\_atoms, 3)`.

    """

    lattice = lattice_params_to_matrix_mindspore(lengths, angles)
    lattice_nodes = lift_globaltonode(lattice, batch)
    pos = einsum_bibijbj(frac_coords, lattice_nodes)

    return pos


def einsum_bibijbj(input_1, input_2):
    """
    ops.einsum is not support on Ascend,
    Here only deal with bi bij -> bj

    Args:
        input_1 (Tensor): The shape of Tensor (b, i)
        input_2 (Tensor): The shape of Tensor (b, i, j)

    Returns:
        (Tensor): The shape of Tensor (b, j)
    """
    input_1 = input_1.view(-1, 1, 3)
    result = mint.bmm(input_1, input_2)

    result_squeezed = ms.ops.squeeze(result, axis=1)
    return result_squeezed


def cart_to_frac_coords(
        cart_coords,
        lengths,
        angles,
        batch,
        lift_globaltonode):
    """ Turned the pos from cart coords to frac coords

    Args:
        cart_coords (Tensor): cart position of each atom for each crystal
        lengths (Tensor): a, b, c for each cystal
        angles (Tensor): alpha, beta, gamma for each cystal
        num_atoms (Tensor): number of atoms for each cystal

    return:
        frac_coords (Tensor): frac coords of each atom for each crystal
    """

    lattice = lattice_params_to_matrix_mindspore(lengths, angles)
    inv_lattice = mint.linalg.inv(lattice)
    inv_lattice_nodes = lift_globaltonode(inv_lattice, batch)
    frac_coords = einsum_bibijbj(cart_coords, inv_lattice_nodes)
    return frac_coords % 1.


def min_distance_sqr_pbc(cart_coords1, cart_coords2, lengths, angles,
                         batch, batch_size, lift_globaltonode, return_vector=False,
                         return_to_jimages=False):
    r"""Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.

    Args:
        cart_coords1 (Tensor): The shape of tensor is :math:`(total\_atoms, 3)`.
        cart_coords2 (Tensor): The shape of tensor is :math:`(total\_atoms, 3)`.
        lengths (Tensor): The shape of tensor is :math:`(total\_atoms, 3)`.
        angles (Tensor): The shape of tensor is :math:`(total\_atoms, 3)`.
        batch (Tensor): The shape of tensor is :math:`(total\_atoms)`.
        batch_size (int): Batch_size.
        lift_globaltonode (function): The lift function used.
        return_vector (bool): If return vector: True. Default: False.
        return_to_jimages (bool): If return to_jimages: True. Default: False.

    Returns:
        min_atom_distance_sqr (Tensor): The shape of tensor is :math:`(total\_atoms, 3, )
        min_atom_distance_vector (Tensor, Optional): Return in `return_vector=True`.
        Vector pointing from cart_coords1 to cart_coords2. The shape of tensor is :math:`(total\_atoms, 3)`.
        to_jimages (Tensor, Optional): Return if `return_to_jimage=True`.
        position of cart_coord2 relative to cart_coord1 in pbc. The shape of tensor is :math:`(total\_atoms, 3)`.
    """
    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = ms.Tensor(OFFSET_LIST, ms.float32)
    num_cells = unit_cell.shape[0]
    unit_cell = mint.permute(unit_cell, (1, 0))
    unit_cell_batch = mint.broadcast_to(unit_cell.view(1, 3, num_cells),
                                        (batch_size, -1, -1))

    # lattice matrix
    lattice = lattice_params_to_matrix_mindspore(lengths, angles)
    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = mint.permute(lattice, (0, 2, 1))
    pbc_offsets = mint.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = lift_globaltonode(pbc_offsets, batch)
    # Expand the positions and indices for the 9 cells
    pos1 = mint.broadcast_to(pos1.view(-1, 3, 1), (-1, -1, num_cells))
    pos2 = mint.broadcast_to(pos2.view(-1, 3, 1), (-1, -1, num_cells))
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom
    # Compute the vector between atoms
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = mint.sum(ms.ops.pow(atom_distance_vector, 2), dim=1)
    min_atom_distance_sqr, min_indices = mint.min(atom_distance_sqr, dim=1)
    return_list = []
    return_list.append(min_atom_distance_sqr)
    if return_vector:
        min_indices = mint.tile(min_indices.view(-1, 1, 1), (1, 3, 1))

        min_atom_distance_vector = ms.ops.gather_elements(
            atom_distance_vector, 2, min_indices).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = mint.index_select(unit_cell.T, 0, min_indices)
        return_list.append(to_jimages)
    return return_list[0] if len(return_list) == 1 else return_list


class StandardScalerMindspore:
    """
    Normalizes the targets of a dataset. (Mindpsore versrion of StandardScaler,
    some functions might be different.)
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.

    Args:
        means (numpy.ndarray): An optional 1D numpy array of precomputed means. Default: None.
        stds (numpy.ndarray): An optional 1D numpy array of precomputed standard deviations.
            Default: None.
        replace_nan_token (float): A token to use to replace NaN entries in the features.
            Default: None.
    """

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )

    def fit(self, x):

        self.means = mint.mean(x, axis=0)
        eps = 1e-5
        mean_squared_diff = ms.ops.mse_loss(x, self.means)
        std_dev = mint.sqrt(mean_squared_diff)
        self.stds = std_dev + eps

    def transform(self, x):
        return mint.div((x - self.means), self.stds)

    def inverse_transform(self, x):
        return mint.add(mint.mul(x, self.stds), self.means)


def get_scaler_from_data_list(data_list, key):
    """
    get scaler from data_list

    Args:
        data_list (numpy.ndarray): The data list.
        key (numpy.ndarray): The key of data list.

    Returns:
        scaler (numpy.ndarray):
    """
    targets = [d[key] for d in data_list]
    scaler = StandardScaler()
    scaler.fit(targets)
    return scaler


class StandardScaler:
    """
    StandardScaler normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.

    Args:
        means (numpy.ndarray): An optional 1D numpy array of precomputed means. Default: None.
        stds (numpy.ndarray): An optional 1D numpy array of precomputed standard deviations.
            Default: None.
        replace_nan_token (float): A token to use to replace NaN entries in the features.
            Default: None.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, x):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        Args:
            x (numpy.ndarray): A list of lists of floats.
        Returns:
            The fitted class:`StandardScaler` (self).
        """
        self.means = np.nanmean(x, axis=0)
        self.stds = np.nanstd(x, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, x):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        Args:
            x (List): A list of lists of floats.

        Returns:
            (numpy.ndarray): The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        x = np.array(x).astype(float)
        transformed_with_nan = (x - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, x):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        Args:
            x: A list of lists of floats.

        Returns:
            The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        x = np.array(x).astype(float)
        transformed_with_nan = x * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def to_mindspore(self):
        """
        Change the type of mean and std to MindSpore Tensor.
        """
        return StandardScalerMindspore(means=ms.Tensor(self.means, ms.float32),
                                       stds=ms.Tensor(self.stds, ms.float32))
