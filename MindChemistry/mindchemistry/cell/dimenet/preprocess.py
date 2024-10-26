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
"""dimenet preprocess"""

import numpy as np
import mindspore as ms

from ..gemnet.utils import SphericalBasisLayer
from ..gemnet.data_utils import frac_to_cart_coords_numpy, get_pbc_distances


class PreProcess:
    """preprocess"""

    def __init__(self, num_spherical, num_radial, envelope_exponent,
                 otf_graph=False, cutoff=10.0, max_num_neighbors=20, task="cdvae"):
        r"""
        Initialize the PreProcess for DimeNetPlusPlus.

        Args:
            num_spherical (int): Number of spherical harmonics.
            num_radial (int): Number of radial basis functions.
            envelope_exponent (float): Exponent for the envelope function.
            otf_graph (bool, optional): Whether to compute the graph on-the-fly. Defaults to False.
            cutoff (float, optional): Cutoff distance for neighbor search. Defaults to 10.0.
            max_num_neighbors (int, optional): Maximum number of neighbors per atom. Defaults to 20.
            task (str, optional): "cdvae" or "dimenet". Defaults to "cdvae".
        """
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.otf_graph = otf_graph
        self.sbf = SphericalBasisLayer(num_spherical, num_radial,
                                       cutoff, envelope_exponent)
        self.task = task

    def data_process(self, angles, lengths, num_atoms, edge_index, frac_coords,
                     num_bonds, to_jimages, atom_types):
        r"""
        Process the input data.

        Args:
            angles (np.ndarray): The shape of tensor is :math:`(batch\_size, 3)`.
            lengths (np.ndarray): The shape of tensor is :math:`(batch\_size, 3)`.
            num_atoms (np.ndarray): The shape of tensor is :math:`(batch\_size,)`.
            edge_index (np.ndarray): The shape of tensor is :math:`(2, total\_edges)`.
            frac_coords (np.ndarray): The shape of tensor is :math:`(total\_atoms, 3)`.
            num_bonds (np.ndarray): The shape of tensor is :math:`(batch\_size,)`.
            to_jimages (np.ndarray): The shape of tensor is :math:`(total\_edges,)`.
            atom_types (np.ndarray): The shape of tensor is :math:`(total\_atoms,)`.

        Returns:
            Tuple[Union[np.ndarray, ms.Tensor]]: A tuple containing the following processed data:
            - **atom_types** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_atoms,)`.
            - **dist** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_edges,)`.
            - **angle** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_edges,)`.
            - **idx_kj** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_triplets,)`.
            - **idx_ji** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_triplets,)`.
            - **edge_j** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_edges,)`.
            - **edge_i** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_edges,)`.
            - **pos** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_atoms, 3)`.
            - **batch** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_atoms,)`.
            - **lengths** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(batch\_size, 3)`.
            - **num_atoms** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(batch\_size,)`.
            - **angles** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(batch\_size, 3)`.
            - **frac_coords** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(total\_atoms, 3)`.
            - **num_bonds** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(batch\_size,)`.
            - **num_triplets** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is :math:`(batch\_size,)`.
            - **sbf** (Union[np.ndarray, ms.Tensor]) - The shape of tensor is
            :math:`(total\_triplets, num\_spherical * num\_radial)`.
        """
        num_atoms = num_atoms.reshape(-1)
        batch = np.repeat(np.arange(num_atoms.shape[0],), num_atoms, axis=0)

        pos = frac_to_cart_coords_numpy(
            frac_coords,
            lengths,
            angles,
            num_atoms)

        out = get_pbc_distances(
            frac_coords,
            edge_index,
            lengths,
            angles,
            to_jimages,
            num_atoms,
            num_bonds,
            return_offsets=True
        )

        edge_index = out["edge_index"]
        dist = out["distances"]
        offsets = out["offsets"]

        edge_j, edge_i = edge_index

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji, num_triplets = self.triplets(
            edge_index)

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_j = pos[idx_j]
        pos_ji, pos_kj = (
            pos[idx_j] - pos_i + offsets[idx_ji],
            pos[idx_k] - pos_j + offsets[idx_kj],
        )

        a = (pos_ji * pos_kj).sum(axis=-1)
        b = np.linalg.norm(np.cross(pos_ji, pos_kj), axis=-1)
        angle = np.arctan2(b, a)
        ####### sbf ######
        sbf = self.sbf.sbf(dist, angle, idx_kj)

        if self.task == "dimenet":
            atom_types = ms.Tensor(atom_types, ms.int32)
            dist = ms.Tensor(dist, ms.float32)
            idx_kj = ms.Tensor(idx_kj, ms.int32)
            idx_ji = ms.Tensor(idx_ji, ms.int32)
            edge_j = ms.Tensor(edge_j, ms.int32)
            edge_i = ms.Tensor(edge_i, ms.int32)
            batch = ms.Tensor(batch, ms.int32)
            sbf = ms.Tensor(sbf, ms.float32)
            return (atom_types, dist, idx_kj, idx_ji, edge_j, edge_i, batch, sbf)

        return (atom_types, dist, angle, idx_kj, idx_ji, edge_j, edge_i, pos,
                batch, lengths, num_atoms, angles, frac_coords, num_bonds, num_triplets, sbf)

    @staticmethod
    def triplets(edge_index):
        """
        Compute triplets from edge_index.

        Args:
            edge_index (np.ndarray): Array of edge indices.

        Returns:
            Tuple: Tuple containing computed triplets.
        """
        node_1, node_2 = edge_index
        num_atoms = np.max(edge_index) + 1
        dict_node_2 = dict()
        idx_ji, idx_kj, idx_i, idx_j, idx_k = [], [], [], [], []
        for j in range(num_atoms):
            dict_node_2[j] = []
            idx = node_2 == j
            ik = node_1[idx]
            idx_edge = np.arange(node_1.shape[0])
            idx_edge = idx_edge[idx]
            dict_node_2[j].append([idx_edge, ik])

        for j in dict_node_2:
            for idx, k in enumerate(dict_node_2[j][0][1]):
                num = len(dict_node_2[j][0][1]) - idx - 1
                i = dict_node_2[j][0][1][idx+1:]
                idx_edge_2 = dict_node_2[j][0][0][idx+1:]
                idx_ji.append([idx] * num)
                idx_kj.append(idx_edge_2)
                idx_i.append([k] * num)
                idx_j.append([j] * num)
                idx_k.append(i)
        idx_ji = np.concatenate(idx_ji).astype(np.int32)
        idx_kj = np.concatenate(idx_kj).astype(np.int32)
        idx_i = np.concatenate(idx_i).astype(np.int32)
        idx_j = np.concatenate(idx_j).astype(np.int32)
        idx_k = np.concatenate(idx_k).astype(np.int32)
        num_triplets = np.bincount(idx_ji, minlength=node_2.shape[0])
        return node_2, node_1, idx_i, idx_j, idx_k, idx_kj, idx_ji, num_triplets
