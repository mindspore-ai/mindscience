# ============================================================================
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
"""Featurization utilities for molecular models."""

from typing import Callable, Optional, Tuple, Union
from pynanoflann import KDTree as NanoKDTree
from scipy.spatial import KDTree as SciKDTree

import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, mint

DistanceFeaturizer = Callable[[Tensor], Tensor]



def gaussian_basis_function(
        scalars: Tensor,
        num_bases: Union[Tensor, int],
        radius: Union[Tensor, float],
        scale: Union[Tensor, float] = 1.0,
) -> Tensor:
    """Gaussian basis function applied to a tensor of scalars.

    Args:
        scalars (Tensor): Scalars to compute the gbf on. Shape [num_scalars].
        num_bases (Tensor): The number of bases. An Int.
        radius (Tensor): The largest centre of the bases. A Float.
        scale (Tensor, optional): The width of the gaussians. Defaults to 1.

    Returns:
        Tensor: A tensor of shape [num_scalars, num_bases].
    """
    assert len(scalars.shape) == 1
    gaussian_means = ops.arange(
        0, float(radius), float(radius) / num_bases
    )
    return mint.exp(
        -(scale**2) * (scalars.unsqueeze(1) - gaussian_means.unsqueeze(0)).abs() ** 2
    )


def featurize_edges(
        edge_vectors: Tensor, distance_featurization: DistanceFeaturizer
) -> Tensor:
    """Featurizes edge features, provides concatenated unit vector along with featurized distances.

    Args:
        edge_vectors (tensor): Edge vectors to featurize. Shape [num_edge, 3]
        distance_featurization (DistanceFeaturization): A function that featurizes the distances of the vectors.

    Returns:
        tensor: Edge features, shape [num_edge, num_edge_features].
    """
    edge_features = []
    edge_norms = mint.linalg.norm(edge_vectors, dim=1)
    featurized_edge_norms = distance_featurization(edge_norms)
    unit_vectors = edge_vectors / edge_norms.unsqueeze(1)
    unit_vectors = mint.nan_to_num(unit_vectors, nan=0, posinf=0, neginf=0)
    edge_features.append(featurized_edge_norms)
    edge_features.append(unit_vectors)
    return mint.cat(edge_features, dim=-1).to(ms.float32)


def compute_edge_vectors(
        edge_index: Tensor, positions: Tensor
) -> Tensor:
    """Computes edge vectors from positions.

    Args:
        edge_index (tensor): The edge index. First position the senders, second
            position the receivers. Shape [2, num_edge].
        positions (tensor): Positions of each node. Shape [num_nodes, 3]

    Returns:
        tensor: The vectors of each edge.
    """
    senders = edge_index[0]
    receivers = edge_index[1]
    return positions[receivers] - positions[senders]


# These are offsets applied to coordinates to create a 3x3x3
# tiled periodic image of the input structure.
OFFSETS = np.array(
    [
        [-1.0, 1.0, -1.0],
        [0.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [-1.0, -1.0, -1.0],
        [0.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [-1.0, -1.0, 1.0],
        [0.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
    ]
)

NUM_OFFSETS = len(OFFSETS)


def _compute_img_positions_torch(
        positions: Tensor, periodic_boundaries: Tensor
) -> Tensor:
    """Computes the positions of the periodic images of the input structure.

    Consider the following 2D periodic boundary image.
    + --- + --- + --- +
    |     |     |     |
    + --- + --- + --- +
    |     |  x  |     |
    + --- + --- + --- +
    |     |     |     |
    + --- + --- + --- +

    Each tile in this has an associated translation to translate
    'x'. For example, the top left would by (-1, +1). These are
    the 'OFFSETS', but OFFSETS are for a 3x3x3 grid.

    This is complicated by the fact that our periodic
    boundaries are not orthogonal to each other, and so we form a new
    translation by taking a linear combination of the unit cell axes.

    Args:
        positions (Tensor): Positions of the atoms. Shape [num_atoms, 3].
        periodic_boundaries (Tensor): Periodic boundaries of the unit cell.
            This can be 2 shapes - [3, 3] or [num_atoms, 3, 3]. If the shape is
            [num_atoms, 3, 3], it is assumed that the PBC has been repeat_interleaved
            for each atom, i.e this function is agnostic as to whether it is computing
            with respect to a batch or not.
    Returns:
        Tensor: The positions of the periodic images. Shape [num_atoms, 27, 3].
    """
    num_positions = len(positions)

    has_unbatched_pbc = periodic_boundaries.shape == (3, 3)
    if has_unbatched_pbc:
        periodic_boundaries = periodic_boundaries.unsqueeze(0)
        periodic_boundaries = periodic_boundaries.expand((num_positions, 3, 3))

    assert periodic_boundaries.shape[0] == positions.shape[0]
    offsets = Tensor(OFFSETS, dtype=positions.dtype)
    offsets = mint.unsqueeze(offsets, 0)
    repeated_offsets = offsets.expand((num_positions, NUM_OFFSETS, 3))
    repeated_offsets = mint.unsqueeze(repeated_offsets, 3)
    periodic_boundaries = mint.unsqueeze(periodic_boundaries, 1)
    translations = repeated_offsets * periodic_boundaries
    translations = translations.sum(2)

    # Expand the positions so we can broadcast add the translations per PBC image.
    expanded_positions = positions.unsqueeze(1)
    translated_positions = expanded_positions + translations
    return translated_positions


def brute_force_knn(
        img_positions: Tensor, positions: Tensor, k: int
) -> Tuple[Tensor, Tensor]:
    """Brute force k-nearest neighbors.

    Args:
        img_positions (Tensor): The positions of the images. Shape [num_atoms * 27, 3].
        positions (Tensor): The positions of the query atoms. Shape [num_atoms, 3].
        k (int): The number of nearest neighbors to find.

    Returns:
        return_types.topk: The indices of the nearest neighbors. Shape [num_atoms, k].
    """
    dist = mint.cdist(positions, img_positions)
    return mint.topk(dist, k, largest=False, sorted=True)


def compute_pbc_radius_graph(
        *,
        positions: Tensor,
        periodic_boundaries: Tensor,
        radius: Union[float, Tensor],
        max_number_neighbors: int = 20,
        brute_force: Optional[bool] = None,
        library: str = "pynanoflann",
        n_workers: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Computes periodic condition radius graph from positions.

    Args:
        positions (Tensor): 3D positions of particles. Shape [num_particles, 3].
        periodic_boundaries (Tensor): A 3x3 matrix where the periodic boundary axes are rows or columns.
        radius (Union[float, tensor]): The radius within which to connect atoms.
        max_number_neighbors (int, optional): The maximum number of neighbors for each particle. Defaults to 20.
        brute_force (bool, optional): Whether to use brute force knn. Defaults to None, in which case brute_force
            is used if GPU is available (2-6x faster), but not on CPU (1.5x faster - 4x slower, depending on
            system size).
        library (str, optional): The KDTree library to use. Currently, either 'scipy' or 'pynanoflann'.
        n_workers (int, optional): The number of workers to use for KDTree construction. Defaults to 1.

    Returns:
        Tuple[Tensor, Tensor]: A 2-Tuple. First, an edge_index tensor, where the first index are the
        sender indices and the second are the receiver indices. Second, the vector displacements between edges.
    """
    if brute_force is None:
        brute_force = ms.get_context("device_target") == "GPU"

    if mint.any(periodic_boundaries != 0.0):
        supercell_positions = _compute_img_positions_torch(
            positions=positions, periodic_boundaries=periodic_boundaries
        )
        # CRITICALLY IMPORTANT: We need to reshape the supercell_positions to be
        # flat, so we can use them for the nearest neighbors. The *way* in which
        # they are flattened is important, because we need to be able to map the
        # indices returned from the nearest neighbors to the original positions.
        # The easiest way to do this is to transpose, so that when we flatten, we
        # have:
        # [
        #   img_0_atom_0,
        #   img_0_atom_1,
        #   ...,
        #   img_0_atom_N,
        #   img_1_atom_0,
        #   img_1_atom_1,
        #   ...,
        #   img_N_atom_N,
        #   etc
        # ]
        # This way, we can take the mod of the indices returned from the nearest
        # neighbors to get the original indices.
        # Shape (27, num_positions, 3)
        supercell_positions = supercell_positions.transpose(0, 1)
        supercell_positions = supercell_positions.reshape(-1, 3)
    else:
        supercell_positions = positions

    num_positions = positions.shape[0]

    if brute_force:
        # Brute force
        distance_values, nearest_img_neighbors = brute_force_knn(
            supercell_positions,
            positions,
            min(max_number_neighbors + 1, len(supercell_positions)),
        )

        # remove distances greater than radius, and exclude self
        within_radius = distance_values[:, 1:] < (radius + 1e-6)

        num_neighbors_per_position = within_radius.sum(-1)
        # remove the self node which will be closest
        index_array = nearest_img_neighbors[:, 1:]

        senders = mint.repeat_interleave(
            mint.arange(num_positions), num_neighbors_per_position
        )
        receivers_imgs = index_array[within_radius]

        receivers = receivers_imgs % num_positions
        vectors = supercell_positions[receivers_imgs] - positions[senders]
        stacked = mint.stack((senders, receivers), dim=0)
        return stacked, vectors

    # Build a KDTree from the supercell positions.
    # Query that KDTree just for the positions in the central cell.
    tree_data = supercell_positions.clone().numpy()
    tree_query = positions.clone().numpy()
    distance_upper_bound = np.array(radius) + 1e-8
    if library == "scipy":
        tree = SciKDTree(tree_data, leafsize=100)
        _, nearest_img_neighbors = tree.query(
            tree_query,
            max_number_neighbors + 1,
            distance_upper_bound=distance_upper_bound,
            workers=n_workers,
            p=2,
        )
        # Remove the self-edge that will be closest
        index_array = np.array(nearest_img_neighbors)[:, 1:]
        # Remove any entry that equals len(supercell_positions), which are negative hits
        receivers_imgs = index_array[index_array != len(supercell_positions)]
        num_neighbors_per_position = (index_array != len(supercell_positions)).sum(
            -1
        )
    elif library == "pynanoflann":
        tree = NanoKDTree(
            n_neighbors=min(max_number_neighbors + 1, len(supercell_positions)),
            radius=radius,
            leaf_size=100,
            metric="l2",
        )
        tree.fit(tree_data)
        distance_values, nearest_img_neighbors = tree.kneighbors(
            tree_query, n_jobs=n_workers
        )
        nearest_img_neighbors = nearest_img_neighbors.astype(np.int32)

        # remove the self node which will be closest
        index_array = nearest_img_neighbors[:, 1:]
        # remove distances greater than radius
        within_radius = distance_values[:, 1:] < (radius + 1e-6)
        receivers_imgs = index_array[within_radius]
        num_neighbors_per_position = within_radius.sum(-1)

    # We construct our senders and receiver indexes.
    senders = np.repeat(np.arange(num_positions), list(num_neighbors_per_position))
    receivers_img_torch = Tensor(receivers_imgs, ms.int32)
    # Map back to indexes on the central image.
    receivers = receivers_img_torch % num_positions
    senders_torch = Tensor(senders, ms.int32)

    # Finally compute the vector displacements between senders and receivers.
    vectors = supercell_positions[receivers_img_torch] - positions[senders_torch]
    return mint.stack((senders_torch, receivers), dim=0), vectors


def batch_map_to_pbc_cell(
        positions: Tensor,
        periodic_boundary_conditions: Tensor,
        num_atoms: Tensor,
) -> Tensor:
    """Maps positions to within a periodic boundary cell, for a batched system.

    Args:
        positions (Tensor): The positions to be mapped. Shape [num_particles, 3]
        periodic_boundary_conditions (Tensor): The periodic boundary conditions. Shape [num_batches, 3, 3]
        num_atoms (LongTensor): The number of atoms in each batch. Shape [num_batches]
    """
    dtype = positions.dtype
    positions = positions.double()
    periodic_boundary_conditions = periodic_boundary_conditions.double()

    pbc_nodes = mint.repeat_interleave(periodic_boundary_conditions, num_atoms, dim=0)

    # To use the stable linalg.solve, we need to mask batch elements which don't
    # have periodic boundaries. We do this by adding the identity matrix as their PBC,
    # because we need the PBCs to be non-singular.
    null_pbc = pbc_nodes.abs().sum(dim=[1, 2]) == 0
    identity = mint.eye(3, dtype=ms.bool_)
    # Broadcast the identity to the elements of the batch that have a null pbc.
    null_pbc_identity_mask = null_pbc.view(-1, 1, 1) & identity.view(1, 3, 3)
    pbc_nodes_masked = pbc_nodes + null_pbc_identity_mask.double()

    lattice_coords = ops.matrix_solve(pbc_nodes_masked.transpose(1, 2), positions)
    frac_coords = lattice_coords % 1.0

    cartesian = mint.einsum("bi,bij->bj", frac_coords, pbc_nodes)
    return mint.where(null_pbc.unsqueeze(1), positions, cartesian).to(dtype)


def batch_compute_pbc_radius_graph(
        *,
        positions: Tensor,
        periodic_boundaries: Tensor,
        radius: Union[float, Tensor],
        image_idx: Tensor,
        max_number_neighbors: int = 20,
        brute_force: Optional[bool] = None,
        library: str = "scipy",
):
    """Computes batched periodic boundary condition radius graph from positions.

    This function is optimised for computation on CPU, and work work on device. GPU implementations
    are likely to be significantly slower because of the irregularly sized tensor computations and the
    lack of extremely fast GPU knn routines.

    Args:
        positions (Tensor): 3D positions of a batch of particles. Shape [num_particles, 3].
        periodic_boundaries (Tensor): A batch where each element 3x3 matrix where the periodic boundary axes
            are rows or columns.
        radius (Union[float, tensor]): The radius within which to connect atoms.
        image_idx (Tensor): A vector where each element indicates the number of particles in each element of
            the batch. Of size len(batch).
        max_number_neighbors (int, optional): The maximum number of neighbors for each particle. Defaults to 20.
        brute_force (bool, optional): Whether to use brute force knn. Defaults to None, in which case brute_force
            is used if we are on GPU (2-6x faster), but not on CPU (1.5x faster - 4x slower).
        library (str, optional): The KDTree library to use. Currently, either 'scipy' or 'pynanoflann'.

    Returns:
        Tuple[Tensor, Tensor]: A 2-Tuple. First, an edge_index tensor, where the first index are the
        sender indices and the second are the receiver indices. Second, the vector displacements between edges.
    """
    idx = 0
    all_edges = []
    all_vectors = []
    num_edges = []

    for p, pbc in zip(
            ops.tensor_split(positions, mint.cumsum(image_idx, 0)[:-1]),
            periodic_boundaries,
    ):
        edges, vectors = compute_pbc_radius_graph(
            positions=p,
            periodic_boundaries=pbc,
            radius=radius,
            max_number_neighbors=max_number_neighbors,
            brute_force=brute_force,
            library=library,
        )
        if idx == 0:
            offset = 0
        else:
            offset += image_idx[idx - 1]
        all_edges.append(edges + offset)
        all_vectors.append(vectors)
        num_edges.append(len(edges[0]))
        idx += 1

    all_edges = ms.numpy.concatenate(all_edges, 1)
    all_vectors = ms.numpy.concatenate(all_vectors, 0)
    num_edges = Tensor(num_edges, dtype=ms.int64)
    return all_edges, all_vectors, num_edges
