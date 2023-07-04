# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
transform
"""

import copy
import logging
from collections import deque
import mindspore as ms
from mindspore import numpy as ops


logger = logging.getLogger(__name__)


class NormalizeTarget:
    """
    Normalize the target values in a sample.
    Parameters:
        mean (dict of float): mean of targets
        std (dict of float): standard deviation of targets
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, item):
        item = item.copy()
        for k in self.mean:
            if k in item:
                item[k] = (item[k] - self.mean[k]) / self.std[k]
            else:
                raise ValueError(f"Can't find target `{k}` in data item")
        return item


class RemapAtomType:
    """
    Map atom types to their index in a vocabulary. Atom types that don't present in the vocabulary are mapped to -1.
    Parameters:
        atom_types (array_like): vocabulary of atom types
    """

    def __init__(self, atom_types):
        atom_types = ms.Tensor(atom_types)
        self.id2atom = atom_types
        self.atom2id = - ops.ones(atom_types.max() + 1, dtype=ms.int64)
        self.atom2id[atom_types] = ops.arange(len(atom_types))

    def __call__(self, item):
        graph = copy.copy(item["graph"])
        graph.atom_type = self.atom2id[graph.atom_type]
        item = item.copy()
        item["graph"] = graph
        return item


class RandomBFSOrder:
    """
    Order the nodes in a graph according to a random BFS order.
    """
    @staticmethod
    def __call__(item):
        graph = item["graph"]
        edges = graph.edges.tolist()
        neighbor = [[] for _ in range(graph.n_node)]
        for h, t in edges:
            neighbor[h].append(t)
        depth = [-1] * graph.num_node

        i = int(ops.randint(graph.n_node, (1,)))
        queue = deque([i])
        depth[i] = 0
        order = []
        while queue:
            h = queue.popleft()
            order.append(h)
            for t in neighbor[h]:
                if depth[t] == -1:
                    depth[t] = depth[h] + 1
                    queue.append(t)

        item = item.copy()
        item["graph"] = graph.subgraph(order)
        return item


class Shuffle:
    """
    Shuffle the order of nodes and edges in a graph.
    Parameters:
        shuffle_node (bool, optional): shuffle node order or not
        shuffle_edge (bool, optional): shuffle edge order or not
    """

    def __init__(self, shuffle_node=True, shuffle_edge=True):
        self.shuffle_node = shuffle_node
        self.shuffle_edge = shuffle_edge

    def __call__(self, item):
        graph = item["graph"]
        data = self.transform_data(graph.data_dict, graph.meta)

        item = item.copy()
        item["graph"] = type(graph)(**data)
        return item

    def transform_data(self, data, meta):
        """_summary_

        Args:
            data (_type_): _description_
            meta (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_node = data.n_node
        n_edge = data.n_edge
        if self.shuffle_edge:
            node_perm = ops.randperm(n_node)
        else:
            node_perm = ops.arange(n_node)
        if self.shuffle_edge:
            edge_perm = ops.randperm(n_edge)
        else:
            edge_perm = ops.randperm(n_edge)
        new_data = {}
        for key in data:
            if meta[key] == "node":
                new_data[key] = data[key][node_perm]
            elif meta[key] == "edge":
                new_data[key] = node_perm[data[key][edge_perm]]
            else:
                new_data[key] = data[key]

        return new_data


class VirtualNode:
    """
    Add a virtual node and connect it with every node in the graph.
    Parameters:
        relation (int, optional): relation of virtual edges.
            By default, use the maximal relation in the graph plus 1.
        weight (int, optional): weight of virtual edges
        node_feature (array_like, optional): feature of the virtual node
        edge_feature (array_like, optional): feature of virtual edges
        kwargs: other attributes of the virtual node or virtual edges
    """

    def __init__(self, relation=None, weight=1, node_feat=None, edge_feat=None, **kwargs):
        self.relation = relation
        self.weight = weight

        self.default = {k: ms.Tensor(v) for k, v in kwargs.items()}
        if node_feat is not None:
            self.default["node_feat"] = ms.Tensor(node_feat)
        if edge_feat is not None:
            self.default["edge_feat"] = ms.Tensor(edge_feat)

    def __call__(self, item):
        graph = item["graph"]
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        num_node = graph.num_node
        num_relation = graph.num_relation

        existing_node = ops.arange(num_node)
        virtual_node = ops.ones(num_node, dtype=ops.int64) * num_node
        node_in = ops.concat([virtual_node, existing_node])
        node_out = ops.concat([existing_node, virtual_node])
        if edge_list.shape[1] == 2:
            new_edge = ops.stack([node_in, node_out], axis=-1)
        else:
            if self.relation is None:
                relation = num_relation
                num_relation = num_relation + 1
            else:
                relation = self.relation
            relation = relation * ops.ones(num_node * 2, dtype=ms.int64)
            new_edge = ops.stack([node_in, node_out, relation], axis=-1)
        edge_list = ops.concat([edge_list, new_edge])
        new_edge_weight = self.weight * ops.ones(num_node * 2)
        edge_weight = ops.concat([edge_weight, new_edge_weight])

        # add default node/edge attributes
        data = graph.data_dict.copy()
        for key, value in graph.meta.items():
            if value == "node":
                if key in self.default:
                    new_data = self.default[key].unsqueeze(0)
                else:
                    new_data = ops.zeros(1, *data[key].shape[1:], dtype=data[key].dtype)
                data[key] = ops.concat([data[key], new_data])
            elif value == "edge":
                if key in self.default:
                    repeat = [-1] * (data[key].ndim - 1)
                    new_data = self.default[key].expand(num_node * 2, *repeat)
                else:
                    new_data = ops.zeros(num_node * 2, *data[key].shape[1:],
                                         dtype=data[key].dtype)
                data[key] = ops.concat([data[key], new_data])

        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=num_node + 1,
                            num_relation=num_relation, meta=graph.meta, **data)

        item = item.copy()
        item["graph"] = graph
        return item


class VirtualAtom(VirtualNode):
    """
    Add a virtual atom and connect it with every atom in the molecule.
    Parameters:
        atom_type (int, optional): type of the virtual atom
        bond_type (int, optional): type of the virtual bonds
        node_feat (array_like, optional): feature of the virtual atom
        edge_feat (array_like, optional): feature of virtual bonds
        kwargs: other attributes of the virtual atoms or virtual bonds
    """

    def __init__(self, atom_type=None, bond_type=None, node_feat=None, edge_feat=None, **kwargs):
        super().__init__(relation=bond_type, weight=1, node_feat=node_feat,
                         edge_feat=edge_feat, atom_type=atom_type, **kwargs)


class TruncateProtein:
    """
    Truncate over long protein sequences into a fixed length.
    Parameters:
        max_length (int, optional): maximal length of the sequence. Truncate the sequence if it exceeds this limit.
        random (bool, optional): truncate the sequence at a random position.
            If not, truncate the suffix of the sequence.
        keys (str or list of str, optional): keys for the items that require truncation in a sample
    """

    def __init__(self, max_length=None, random=False, keys="graph"):
        self.truncate_length = max_length
        self.random = random
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def __call__(self, item):
        new_item = item.copy()
        for key in self.keys:
            graph = item[key]
            if graph.num_residue > self.truncate_length:
                if self.random:
                    start = int(ops.randint(graph.num_residue - self.truncate_length, (1,)))
                else:
                    start = 0
                end = start + self.truncate_length
                mask = ops.zeros(graph.num_residue, dtype=ms.int8)
                mask[start:end] = True
                graph = graph.subresidue(mask)

            new_item[key] = graph
        return new_item


class ProteinView:
    """
    Convert proteins to a specific view.
    Parameters:
        view (str): protein view. Can be ``atom`` or ``residue``.
        keys (str or list of str, optional): keys for the items that require view change in a sample
    """

    def __init__(self, view, keys="graph"):
        self.view = view
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def __call__(self, item):
        item = item.copy()
        for key in self.keys:
            graph = copy.copy(item[key])
            graph.view = self.view
            item[key] = graph
        return item


class Compose:
    """
    Compose a list of transforms into one.
    Parameters:
        transforms (list of callable): list of transforms
    """

    def __init__(self, transforms):
        # flatten recursive composition
        new_transforms = []
        for transform in transforms:
            if isinstance(transform, Compose):
                new_transforms += transform.transforms
            elif transform is not None:
                new_transforms.append(transform)
        self.transforms = new_transforms

    def __call__(self, item):
        for transform in self.transforms:
            item = transform(item)
