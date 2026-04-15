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
"""dataloader
"""
import random
import numpy as np
from mindspore import Tensor
import mindspore as ms


class DataLoaderBase:
    r"""
    DataLoader that stacks a batch of graph data to fixed-size Tensors

    For specific dataset, usually the following functions should be customized to include different fields:
    __init__, shuffle_action, __iter__

    """

    def __init__(self,
                 batch_size,
                 edge_index,
                 label=None,
                 node_attr=None,
                 edge_attr=None,
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

        ### can be customized to specific dataset
        self.label = label
        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.sample_num = len(self.node_attr)
        batch_size_div = self.batch_size
        if batch_size_div != 0:
            self.step_num = int(self.sample_num / batch_size_div)
        else:
            raise ValueError

        if dynamic_batch_size:
            self.max_start_sample = self.sample_num
        else:
            self.max_start_sample = self.sample_num - self.batch_size + 1

        self.set_global_max_node_edge_num(self.node_attr, self.edge_attr, max_node, max_edge, shuffle_dataset,
                                          dynamic_batch_size)
        #######

    def __len__(self):
        return self.sample_num

    ### example of generating data of each step, can be customized to specific dataset
    def __iter__(self):
        if self.shuffle_dataset:
            self.shuffle()
        else:
            self.restart()

        while self.index < self.max_start_sample:
            # pylint: disable=W0612
            edge_index_step, node_batch_step, node_mask, edge_mask, batch_size_mask, node_num, edge_num, batch_size \
                = self.gen_common_data(self.node_attr, self.edge_attr)

            ### can be customized to generate different attributes or labels according to specific dataset
            node_attr_step = self.gen_node_attr(self.node_attr, batch_size, node_num)
            edge_attr_step = self.gen_edge_attr(self.edge_attr, batch_size, edge_num)
            label_step = self.gen_global_attr(self.label, batch_size)

            self.add_step_index(batch_size)

            ### make number to Tensor, if it is used as a Tensor in the network
            node_num = Tensor(node_num)
            batch_size = Tensor(batch_size)

            yield node_attr_step, edge_attr_step, label_step, edge_index_step, node_batch_step, \
                node_mask, edge_mask, node_num, batch_size

    @staticmethod
    def pad_zero_to_end(src, axis, zeros_len):
        """pad_zero_to_end"""
        pad_shape = []
        for i in range(src.ndim):
            if i == axis:
                pad_shape.append((0, zeros_len))
            else:
                pad_shape.append((0, 0))
        return np.pad(src, pad_shape)

    @staticmethod
    def gen_mask(total_len, real_len):
        """gen_mask"""
        mask = np.concatenate((np.full((real_len,), np.float32(1)), np.full((total_len - real_len,), np.float32(0))))
        return mask

    ### example of computing global max length of node_attr and edge_attr, can be customized to specific dataset
    def set_global_max_node_edge_num(self,
                                     node_attr,
                                     edge_attr,
                                     max_node=None,
                                     max_edge=None,
                                     shuffle_dataset=True,
                                     dynamic_batch_size=True):
        """set_global_max_node_edge_num

        Args:
            node_attr: node_attr
            edge_attr: edge_attr
            max_node: max_node. Defaults to None.
            max_edge: max_edge. Defaults to None.
            shuffle_dataset: shuffle_dataset. Defaults to True.
            dynamic_batch_size: dynamic_batch_size. Defaults to True.

        Raises:
            ValueError: ValueError
        """
        if not shuffle_dataset:
            max_node_num, max_edge_num = self.get_max_node_edge_num(node_attr, edge_attr, dynamic_batch_size)
            self.max_node_num_global = max_node_num if max_node is None else max(max_node, max_node_num)
            self.max_edge_num_global = max_edge_num if max_edge is None else max(max_edge, max_edge_num)
            return

        sum_node = 0
        sum_edge = 0
        count = 0
        max_node_single = 0
        max_edge_single = 0
        for step in range(self.sample_num):
            node_len = len(node_attr[step])
            edge_len = len(edge_attr[step])
            sum_node += node_len
            sum_edge += edge_len
            max_node_single = max(max_node_single, node_len)
            max_edge_single = max(max_edge_single, edge_len)
            count += 1
        if count != 0:
            mean_node = sum_node / count
            mean_edge = sum_edge / count
        else:
            raise ValueError

        if max_node is not None and max_edge is not None:
            if max_node < max_node_single:
                raise ValueError(
                    f"the max_node {max_node} is less than the max length of a single sample {max_node_single}")
            if max_edge < max_edge_single:
                raise ValueError(
                    f"the max_edge {max_edge} is less than the max length of a single sample {max_edge_single}")

            self.max_node_num_global = max_node
            self.max_edge_num_global = max_edge
        elif max_node is None and max_edge is None:
            sum_node = 0
            sum_edge = 0
            for step in range(self.sample_num):
                sum_node += (len(node_attr[step]) - mean_node) ** 2
                sum_edge += (len(edge_attr[step]) - mean_edge) ** 2

            if count != 0:
                std_node = np.sqrt(sum_node / count)
                std_edge = np.sqrt(sum_edge / count)
            else:
                raise ValueError

            self.max_node_num_global = int(self.batch_size * mean_node +
                                           self.padding_std_ratio * np.sqrt(self.batch_size) * std_node)
            self.max_edge_num_global = int(self.batch_size * mean_edge +
                                           self.padding_std_ratio * np.sqrt(self.batch_size) * std_edge)
            self.max_node_num_global = max(self.max_node_num_global, max_node_single)
            self.max_edge_num_global = max(self.max_edge_num_global, max_edge_single)
        elif max_node is None:
            if max_edge < max_edge_single:
                raise ValueError(
                    f"the max_edge {max_edge} is less than the max length of a single sample {max_edge_single}")

            if mean_edge != 0:
                self.max_node_num_global = int(max_edge * mean_node / mean_edge)
            else:
                raise ValueError
            self.max_node_num_global = max(self.max_node_num_global, max_node_single)
            self.max_edge_num_global = max_edge
        else:
            if max_node < max_node_single:
                raise ValueError(
                    f"the max_node {max_node} is less than the max length of a single sample {max_node_single}")

            self.max_node_num_global = max_node
            if mean_node != 0:
                self.max_edge_num_global = int(max_node * mean_edge / mean_node)
            else:
                raise ValueError
            self.max_edge_num_global = max(self.max_edge_num_global, max_edge_single)

    def get_max_node_edge_num(self, node_attr, edge_attr, remainder=True):
        """get_max_node_edge_num

        Args:
            node_attr: node_attr
            edge_attr: edge_attr
            remainder (bool, optional): remainder. Defaults to True.

        Returns:
            max_node_num, max_edge_num
        """
        max_node_num = 0
        max_edge_num = 0
        index = 0
        for _ in range(self.step_num):
            node_num = 0
            edge_num = 0
            for _ in range(self.batch_size):
                node_num += len(node_attr[index])
                edge_num += len(edge_attr[index])
                index += 1
            max_node_num = max(max_node_num, node_num)
            max_edge_num = max(max_edge_num, edge_num)

        if remainder:
            remain_num = self.sample_num - index - 1
            node_num = 0
            edge_num = 0
            for _ in range(remain_num):
                node_num += len(node_attr[index])
                edge_num += len(edge_attr[index])
                index += 1
            max_node_num = max(max_node_num, node_num)
            max_edge_num = max(max_edge_num, edge_num)

        return max_node_num, max_edge_num

    def shuffle_index(self):
        """shuffle_index"""
        indices = list(range(self.sample_num))
        random.shuffle(indices)
        return indices

    ### example of shuffling the input dataset, can be customized to specific dataset
    def shuffle_action(self):
        """shuffle_action"""
        indices = self.shuffle_index()
        self.edge_index = [self.edge_index[i] for i in indices]
        self.label = [self.label[i] for i in indices]
        self.node_attr = [self.node_attr[i] for i in indices]
        self.edge_attr = [self.edge_attr[i] for i in indices]

    ### example of generating the final shuffled dataset, can be customized to specific dataset
    def shuffle(self):
        """shuffle"""
        self.shuffle_action()
        if not self.dynamic_batch_size:
            max_node_num, max_edge_num = self.get_max_node_edge_num(self.node_attr, self.edge_attr, remainder=False)
            while max_node_num > self.max_node_num_global or max_edge_num > self.max_edge_num_global:
                self.shuffle_action()
                max_node_num, max_edge_num = self.get_max_node_edge_num(self.node_attr, self.edge_attr, remainder=False)

        self.step = 0
        self.index = 0

    def restart(self):
        """restart"""
        self.step = 0
        self.index = 0

    ### example of calculating dynamic batch size to avoid exceeding the max length of node and edge, can be customized to specific dataset
    def get_batch_size(self, node_attr, edge_attr, start_batch_size):
        """get_batch_size

        Args:
            node_attr: node_attr
            edge_attr: edge_attr
            start_batch_size: start_batch_size

        Returns:
            batch_size
        """
        node_num = 0
        edge_num = 0
        for i in range(start_batch_size):
            index = self.index + i
            node_num += len(node_attr[index])
            edge_num += len(edge_attr[index])

        exceeding = False
        while node_num > self.max_node_num_global or edge_num > self.max_edge_num_global:
            node_num -= len(node_attr[index])
            edge_num -= len(edge_attr[index])
            index -= 1
            exceeding = True
            self.batch_exceeding_num += 1
        if exceeding:
            self.batch_change_num += 1

        return index - self.index + 1

    def gen_common_data(self, node_attr, edge_attr):
        """gen_common_data

        Args:
            node_attr: node_attr
            edge_attr: edge_attr

        Returns:
            common_data
        """
        if self.dynamic_batch_size:
            if self.step >= self.step_num:
                batch_size = self.get_batch_size(node_attr, edge_attr,
                                                 min((self.sample_num - self.index), self.batch_size))
            else:
                batch_size = self.get_batch_size(node_attr, edge_attr, self.batch_size)
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
            edge_index_step = np.concatenate((edge_index_step, self.edge_index[i] + max_edge_index), 1)
            max_edge_index = np.max(edge_index_step) + 1
        edge_num = edge_index_step.shape[1]

        ######################### padding
        edge_index_step = self.pad_zero_to_end(edge_index_step, 1, self.max_edge_num_global - edge_num)
        node_batch_step = self.pad_zero_to_end(node_batch_step, 0, self.max_node_num_global - node_num)

        ######################### mask
        node_mask = self.gen_mask(self.max_node_num_global, node_num)
        edge_mask = self.gen_mask(self.max_edge_num_global, edge_num)
        batch_size_mask = self.gen_mask(self.batch_size, batch_size)

        ######################### make Tensor
        edge_index_step = Tensor(edge_index_step, ms.int32)
        node_batch_step = Tensor(node_batch_step, ms.int32)
        node_mask = Tensor(node_mask)
        edge_mask = Tensor(edge_mask)
        batch_size_mask = Tensor(batch_size_mask)

        return CommonData(edge_index_step, node_batch_step, node_mask, edge_mask, batch_size_mask, node_num, edge_num,
                          batch_size).get_tuple_data()

    def gen_node_attr(self, node_attr, batch_size, node_num):
        """gen_node_attr"""
        node_attr_step = np.concatenate(node_attr[self.index:self.index + batch_size], 0)
        node_attr_step = self.pad_zero_to_end(node_attr_step, 0, self.max_node_num_global - node_num)
        node_attr_step = Tensor(node_attr_step)
        return node_attr_step

    def gen_edge_attr(self, edge_attr, batch_size, edge_num):
        """gen_edge_attr"""
        edge_attr_step = np.concatenate(edge_attr[self.index:self.index + batch_size], 0)
        edge_attr_step = self.pad_zero_to_end(edge_attr_step, 0, self.max_edge_num_global - edge_num)
        edge_attr_step = Tensor(edge_attr_step)
        return edge_attr_step

    def gen_global_attr(self, global_attr, batch_size):
        """gen_global_attr"""
        global_attr_step = np.stack(global_attr[self.index:self.index + batch_size], 0)
        global_attr_step = self.pad_zero_to_end(global_attr_step, 0, self.batch_size - batch_size)
        global_attr_step = Tensor(global_attr_step)
        return global_attr_step

    def add_step_index(self, batch_size):
        """add_step_index"""
        self.index = self.index + batch_size
        self.step += 1

class CommonData:
    """CommonData"""
    def __init__(self, edge_index_step, node_batch_step, node_mask, edge_mask, batch_size_mask, node_num, edge_num,
                 batch_size):
        self.tuple_data = (edge_index_step, node_batch_step, node_mask, edge_mask, batch_size_mask, node_num, edge_num,
                           batch_size)

    def get_tuple_data(self):
        """get_tuple_data"""
        return self.tuple_data
