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
"""dataloader
"""
import random
import numpy as np
from mindspore import Tensor
import mindspore as ms


class DataLoaderBaseCDVAE:
    r"""
    DataLoader for CDVAE
    """

    def __init__(self,
                 batch_size,
                 dataset,
                 shuffle_dataset=True,
                 mode="train"):
        dataset = np.load(
            f"./data/{dataset}/{mode}/processed_data.npy", allow_pickle=True).item()
        self.atom_types = dataset["atom_types"]
        self.dist = dataset["dist"]
        self.angle = dataset["angle"]
        self.idx_kj = dataset["idx_kj"]
        self.idx_ji = dataset["idx_ji"]
        self.edge_j = dataset["edge_j"]
        self.edge_i = dataset["edge_i"]
        self.pos = dataset["pos"]
        self.batch = dataset["batch"]
        self.lengths = dataset["lengths"]
        self.num_atoms = dataset["num_atoms"]
        self.angles = dataset["angles"]
        self.frac_coords = dataset["frac_coords"]
        self.y = dataset["y"]
        self.num_bonds = dataset["num_bonds"]
        self.num_triplets = dataset["num_triplets"]
        self.sbf = dataset["sbf"]
        self.edge_attr = self.edge_j
        self.batch_size = batch_size
        self.index = 0
        self.step = 0
        self.shuffle_dataset = shuffle_dataset
        self.feature = [self.atom_types, self.dist, self.angle, self.idx_kj, self.idx_ji,
                        self.edge_j, self.edge_i, self.pos, self.batch, self.lengths,
                        self.num_atoms, self.angles, self.frac_coords, self.y,
                        self.num_bonds, self.num_triplets, self.sbf]

        # can be customized to specific dataset
        self.label = self.num_atoms
        self.node_attr = self.atom_types
        self.sample_num = len(self.node_attr)

        self.max_start_sample = self.sample_num - self.batch_size + 1

    def get_dataset_size(self):
        return self.sample_num

    def __iter__(self):
        if self.shuffle_dataset:
            self.shuffle()
        else:
            self.restart()
        while self.index < self.max_start_sample:
            # can be customized to generate different attributes or labels according to specific dataset
            num_bonds_step = self.gen_global_attr(
                self.num_bonds, self.batch_size).astype(np.int32)
            num_atoms_step = self.gen_global_attr(
                self.num_atoms, self.batch_size).squeeze().astype(np.int32)
            num_triplets_step = self.gen_global_attr(
                self.num_triplets, self.batch_size).astype(np.int32)
            atom_types_step = self.gen_node_attr(
                self.atom_types, self.batch_size).astype(np.int32)
            dist_step = self.gen_edge_attr(
                self.dist, self.batch_size).astype(np.float32)
            angle_step = self.gen_triplet_attr(
                self.angle, self.batch_size).astype(np.float32)
            idx_kj_step = self.gen_triplet_attr(self.idx_kj, self.batch_size)
            idx_kj_step = self.add_index_offset(
                idx_kj_step, num_bonds_step, num_triplets_step).astype(np.int32)
            idx_ji_step = self.gen_triplet_attr(self.idx_ji, self.batch_size)
            idx_ji_step = self.add_index_offset(
                idx_ji_step, num_bonds_step, num_triplets_step).astype(np.int32)
            edge_j_step = self.gen_edge_attr(self.edge_j, self.batch_size)
            edge_j_step = self.add_index_offset(
                edge_j_step, num_atoms_step, num_bonds_step).astype(np.int32)
            edge_i_step = self.gen_edge_attr(self.edge_j, self.batch_size)
            edge_i_step = self.add_index_offset(
                edge_i_step, num_atoms_step, num_bonds_step).astype(np.int32)
            batch_step = np.repeat(
                np.arange(num_atoms_step.shape[0],), num_atoms_step, axis=0).astype(np.int32)
            lengths_step = self.gen_crystal_attr(
                self.lengths, self.batch_size).astype(np.float32)
            angles_step = self.gen_crystal_attr(
                self.angles, self.batch_size).astype(np.float32)
            frac_coords_step = self.gen_node_attr(
                self.frac_coords, self.batch_size).astype(np.float32)
            y_step = self.gen_global_attr(
                self.y, self.batch_size).astype(np.float32)
            sbf_step = self.gen_triplet_attr(
                self.sbf, self.batch_size).astype(np.float32)
            total_atoms = num_atoms_step.sum().item()
            self.add_step_index(self.batch_size)

            ############## change to mindspore Tensor #############
            yield self.np2tensor(atom_types_step, dist_step, angle_step, idx_kj_step,
                                 idx_ji_step, edge_j_step, edge_i_step, batch_step,
                                 lengths_step, num_atoms_step, angles_step, frac_coords_step,
                                 y_step, self.batch_size, sbf_step, total_atoms)

    def np2tensor(self, atom_types_step, dist_step, angle_step, idx_kj_step,
                  idx_ji_step, edge_j_step, edge_i_step, batch_step,
                  lengths_step, num_atoms_step, angles_step, frac_coords_step,
                  y_step, batch_size, sbf_step, total_atoms):
        """np2tensor"""
        atom_types_step = Tensor(atom_types_step, ms.int32)
        dist_step = Tensor(dist_step, ms.float32)
        angle_step = Tensor(angle_step, ms.float32)
        idx_kj_step = Tensor(idx_kj_step, ms.int32)
        idx_ji_step = Tensor(idx_ji_step, ms.int32)
        edge_j_step = Tensor(edge_j_step, ms.int32)
        edge_i_step = Tensor(edge_i_step, ms.int32)
        batch_step = Tensor(batch_step, ms.int32)
        lengths_step = Tensor(lengths_step, ms.float32)
        num_atoms_step = Tensor(num_atoms_step, ms.int32)
        angles_step = Tensor(angles_step, ms.float32)
        frac_coords_step = Tensor(frac_coords_step, ms.float32)
        y_step = Tensor(y_step, ms.float32)
        sbf_step = Tensor(sbf_step, ms.float32)
        return (atom_types_step, dist_step, angle_step, idx_kj_step,
                idx_ji_step, edge_j_step, edge_i_step, batch_step,
                lengths_step, num_atoms_step, angles_step, frac_coords_step,
                y_step, batch_size, sbf_step, total_atoms)

    def add_index_offset(self, edge_index, num_atoms, num_bonds):
        index_offset = (
            np.cumsum(num_atoms, axis=0) - num_atoms
        )

        index_offset_expand = np.repeat(
            index_offset, num_bonds
        )
        edge_index += index_offset_expand
        return edge_index

    def shuffle_index(self):
        """shuffle_index"""
        indices = list(range(self.sample_num))
        random.shuffle(indices)
        return indices

    def shuffle(self):
        """shuffle"""
        self.shuffle_action()
        self.step = 0
        self.index = 0

    def shuffle_action(self):
        """shuffle_action"""
        indices = self.shuffle_index()
        self.atom_types = [self.atom_types[i] for i in indices]
        self.dist = [self.dist[i] for i in indices]
        self.angle = [self.angle[i] for i in indices]
        self.idx_kj = [self.idx_kj[i] for i in indices]
        self.idx_ji = [self.idx_ji[i] for i in indices]
        self.edge_j = [self.edge_j[i] for i in indices]
        self.edge_i = [self.edge_i[i] for i in indices]
        self.pos = [self.pos[i] for i in indices]
        self.batch = [self.batch[i] for i in indices]
        self.lengths = [self.lengths[i] for i in indices]
        self.num_atoms = [self.num_atoms[i] for i in indices]
        self.angles = [self.angles[i] for i in indices]
        self.frac_coords = [self.frac_coords[i] for i in indices]
        self.y = [self.y[i] for i in indices]
        self.num_bonds = [self.num_bonds[i] for i in indices]
        self.num_triplets = [self.num_triplets[i] for i in indices]
        self.sbf = [self.sbf[i] for i in indices]

    def restart(self):
        """restart"""
        self.step = 0
        self.index = 0

    def gen_node_attr(self, node_attr, batch_size):
        """gen_node_attr"""
        node_attr_step = np.concatenate(
            node_attr[self.index:self.index + batch_size], 0)
        return node_attr_step

    def gen_edge_attr(self, edge_attr, batch_size):
        """gen_edge_attr"""
        edge_attr_step = np.concatenate(
            edge_attr[self.index:self.index + batch_size], 0)

        return edge_attr_step

    def gen_global_attr(self, global_attr, batch_size):
        """gen_global_attr"""
        global_attr_step = np.stack(
            global_attr[self.index:self.index + batch_size], 0)

        return global_attr_step

    def gen_crystal_attr(self, global_attr, batch_size):
        """gen_global_attr"""
        global_attr_step = np.stack(
            global_attr[self.index:self.index + batch_size], 0).squeeze()
        return global_attr_step

    def gen_triplet_attr(self, triplet_attr, batch_size):
        """gen_triplet_attr"""
        global_attr_step = np.concatenate(
            triplet_attr[self.index:self.index + batch_size], 0)

        return global_attr_step

    def add_step_index(self, batch_size):
        """add_step_index"""
        self.index = self.index + batch_size
        self.step += 1
