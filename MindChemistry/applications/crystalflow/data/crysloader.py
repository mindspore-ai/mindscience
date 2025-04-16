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
"""dataloader file"""
import numpy as np
from mindspore import Tensor, ops
import mindspore as ms

from mindchemistry.graph.dataloader import DataLoaderBase, CommonData


class Crysloader(DataLoaderBase):
    """
    Crysloader is used to stacks a batch of graph data to fixed-size Tensors.

    Exactly the same code logic as DataLoaderBase, with additional node attribute 'frac_coords'
    and graph attributes, 'lengths' and 'angles'
    """

    def __init__(self,
                 batch_size,
                 node_attr=None,
                 frac_coords=None,
                 edge_attr=None,
                 edge_index=None,
                 lengths=None,
                 angles=None,
                 lattice_polar=None,
                 num_atoms=None,
                 label=None,
                 padding_std_ratio=3.5,
                 dynamic_batch_size=True,
                 shuffle_dataset=True,
                 max_node=None,
                 max_edge=None):
        self.batch_size = batch_size
        self.edge_index = edge_index
        self.index = 0
        self.step = 0
        self.padding_std_ratio = padding_std_ratio
        self.batch_change_num = 0
        self.batch_exceeding_num = 0
        self.dynamic_batch_size = dynamic_batch_size
        self.shuffle_dataset = shuffle_dataset

        ## can be customized to specific dataset
        self.label = label
        self.node_attr = node_attr
        self.frac_coords = frac_coords
        self.edge_attr = edge_attr
        self.lengths = lengths
        self.angles = angles
        self.lattice_polar = lattice_polar
        self.num_atoms = num_atoms
        self.sample_num = len(self.node_attr)
        batch_size_div = self.batch_size
        if batch_size_div != 0:
            self.step_num = int(self.sample_num / batch_size_div)
        else:
            print('The batch size cannot be set to 0')
            raise ValueError

        if dynamic_batch_size:
            self.max_start_sample = self.sample_num
        else:
            self.max_start_sample = self.sample_num - self.batch_size + 1

        self.set_global_max_node_edge_num(self.node_attr, self.edge_attr,
                                          max_node, max_edge, shuffle_dataset,
                                          dynamic_batch_size)

    def __iter__(self):
        if self.shuffle_dataset:
            self.shuffle()
        else:
            self.restart()

        while self.index < self.max_start_sample:
            edge_index_step, node_batch_step, node_mask, edge_mask, \
                batch_size_mask, node_num, _, batch_size \
                = self.gen_common_data(self.node_attr, self.edge_attr)

            node_attr_step = self.gen_node_attr(self.node_attr, batch_size,
                                                node_num)
            node_attr_step = ops.reshape(node_attr_step, (-1,))
            node_attr_step = ops.Cast()(node_attr_step, ms.int32)
            frac_coords_step = self.gen_node_attr(self.frac_coords, batch_size,
                                                  node_num)
            label_step = self.gen_global_attr(self.label, batch_size)
            lengths_step = self.gen_global_attr(self.lengths, batch_size)
            angles_step = self.gen_global_attr(self.angles, batch_size)
            lattice_polar_step = self.gen_global_attr(self.lattice_polar, batch_size)
            num_atoms_step = self.gen_global_attr(self.num_atoms, batch_size).to(ms.int32)

            self.add_step_index(batch_size)

            ### make number to Tensor, if it is used as a Tensor in the network
            node_num = Tensor(node_num, ms.int32)
            batch_size = Tensor(batch_size, ms.int32)

            yield node_attr_step, frac_coords_step, label_step, lengths_step, \
                angles_step, lattice_polar_step, num_atoms_step, edge_index_step, node_batch_step, \
                node_mask, edge_mask, batch_size_mask, node_num, batch_size

    def shuffle_action(self):
        """shuffle_action"""
        indices = self.shuffle_index()
        self.edge_index = [self.edge_index[i] for i in indices]
        self.label = [self.label[i] for i in indices]
        self.node_attr = [self.node_attr[i] for i in indices]
        self.frac_coords = [self.frac_coords[i] for i in indices]
        self.edge_attr = [self.edge_attr[i] for i in indices]
        self.lengths = [self.lengths[i] for i in indices]
        self.angles = [self.angles[i] for i in indices]
        self.lattice_polar = [self.lattice_polar[i] for i in indices]
        self.num_atoms = [self.num_atoms[i] for i in indices]

    def gen_common_data(self, node_attr, edge_attr):
        """gen_common_data

        Args:
            node_attr: node_attr, i.e. atom types
            edge_attr: edge_attr

        Returns:
            common_data
        """
        if self.dynamic_batch_size:
            if self.step >= self.step_num:
                batch_size = self.get_batch_size(
                    node_attr, edge_attr,
                    min((self.sample_num - self.index), self.batch_size))
            else:
                batch_size = self.get_batch_size(node_attr, edge_attr,
                                                 self.batch_size)
        else:
            batch_size = self.batch_size

        ######################## node_batch
        node_batch_step = []
        sample_num = 0
        for i in range(self.index, self.index + batch_size):
            node_batch_step.extend([sample_num] * node_attr[i].shape[0])
            sample_num += 1
        node_batch_step = np.array(node_batch_step)
        node_num = node_batch_step.shape[0]

        ######################## edge_index
        edge_index_step = np.array([[], []], dtype=np.int64)
        max_edge_index = 0
        for i in range(self.index, self.index + batch_size):
            edge_index_step = np.concatenate(
                (edge_index_step, self.edge_index[i] + max_edge_index), 1)
            max_edge_index = np.max(edge_index_step) + 1
        edge_num = edge_index_step.shape[1]

        ######################### padding
        edge_index_step = self.pad_zero_to_end(
            edge_index_step, 1, self.max_edge_num_global - edge_num)
        node_batch_step = self.pad_zero_to_end(
            node_batch_step, 0, self.max_node_num_global - node_num)

        ######################### mask
        node_mask = self.gen_mask(self.max_node_num_global, node_num)
        edge_mask = self.gen_mask(self.max_edge_num_global, edge_num)
        batch_size_mask = self.gen_mask(self.batch_size, batch_size)

        ######################### make Tensor
        edge_index_step = Tensor(edge_index_step, ms.int32)
        node_batch_step = Tensor(node_batch_step, ms.int32)
        node_mask = Tensor(node_mask, ms.int32)
        edge_mask = Tensor(edge_mask, ms.int32)
        batch_size_mask = Tensor(batch_size_mask, ms.int32)

        return CommonData(edge_index_step, node_batch_step, node_mask,
                          edge_mask, batch_size_mask, node_num, edge_num,
                          batch_size).get_tuple_data()

    def gen_node_attr(self, node_attr, batch_size, node_num):
        """gen_node_attr"""
        node_attr_step = np.concatenate(
            node_attr[self.index:self.index + batch_size], 0)
        node_attr_step = self.pad_zero_to_end(
            node_attr_step, 0, self.max_node_num_global - node_num)
        node_attr_step = Tensor(node_attr_step, ms.float32)
        return node_attr_step

    def gen_edge_attr(self, edge_attr, batch_size, edge_num):
        """gen_edge_attr"""
        edge_attr_step = np.concatenate(
            edge_attr[self.index:self.index + batch_size], 0)
        edge_attr_step = self.pad_zero_to_end(
            edge_attr_step, 0, self.max_edge_num_global - edge_num)
        edge_attr_step = Tensor(edge_attr_step, ms.float32)
        return edge_attr_step

    def gen_global_attr(self, global_attr, batch_size):
        """gen_global_attr"""
        global_attr_step = np.stack(
            global_attr[self.index:self.index + batch_size], 0)
        global_attr_step = self.pad_zero_to_end(global_attr_step, 0,
                                                self.batch_size - batch_size)
        global_attr_step = Tensor(global_attr_step, ms.float32)
        return global_attr_step
