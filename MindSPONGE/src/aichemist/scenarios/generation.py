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
generation
"""
from collections import defaultdict
import copy
import logging
import warnings

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.nn.probability.distribution import Categorical
from tqdm import tqdm

from .. import transforms
from .. import layers
from .. import util
from ..data import MoleculeBatch, Molecule, GraphBatch
from .. import metrics
from .. import core
from ..util import functional
from ..core import Registry as R


logger = logging.getLogger(__name__)


@R.register('scenario.AutogressiveGenerator')
class AutoregressiveGenerator(core.Cell):
    """
    Autoregressive graph generation task.
    This class can be used to implement GraphAF proposed in
    `GraphAF: A Flow-based Autoregressive Model for Molecular Graph Generation`_.
    To do so, instantiate the node model and the edge model with two
    :class:`GraphAutoregressiveFlow <torchdrug.models.GraphAutoregressiveFlow>` models.
    .. _GraphAF: A Flow-based Autoregressive Model for Molecular Graph Generation: https://arxiv.org/pdf/2001.09382.pdf

    Parameters:
        node_model (nn.Cell): node likelihood model
        edge_model (nn.Cell): edge likelihood model
        task (str or list of str, optional): property optimization task(s). Available tasks are ``plogp`` and ``qed``.
        n_node_sample (int, optional): number of node samples per graph. -1 for all samples.
        n_edge_sample (int, optional): number of edge samples per graph. -1 for all samples.
        max_edge_unroll (int, optional): max node id difference.
            If not provided, use the statistics from the training set.
        max_node (int, optional): max number of node.
            If not provided, use the statistics from the training set.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``nll`` and ``ppo``.
        agent_update_interval (int, optional): update agent every n batch
        gamma (float, optional): reward discount rate
        reward_temperature (float, optional): temperature for reward. Higher temperature encourages larger mean reward,
            while lower temperature encourages larger maximal reward.
        baseline_momentum (float, optional): momentum for value function baseline
    """

    eps = 1e-10
    top_k = 10
    _option_members = {"task", "criterion"}

    def __init__(self, node_model, edge_model, task=(), n_node_sample=-1, n_edge_sample=-1,
                 max_edge_unroll=None, max_node=None, criterion="nll", agent_update_interval=5, gamma=0.9,
                 reward_temperature=1, baseline_momentum=0.9):
        super().__init__()
        self.node_model = node_model
        self.edge_model = edge_model
        self.agent_node_model = copy.deepcopy(node_model)
        self.agent_edge_model = copy.deepcopy(edge_model)
        self.task = task
        self.n_atom_type = self.node_model.input_dim
        self.n_bond_type = self.edge_model.input_dim
        self.n_node_sample = n_node_sample
        self.n_edge_sample = n_edge_sample
        self.max_edge_unroll = max_edge_unroll
        self.max_node = max_node
        self.criterion = criterion
        self.agent_update_interval = agent_update_interval
        self.gamma = gamma
        self.reward_temperature = reward_temperature
        self.baseline_momentum = baseline_momentum
        self.best_results = defaultdict(list)
        self.batch_id = 0
        self.node_baseline = None
        self.edge_baseline = None
        self.id2atom = None
        self.atom2id = None

    def preprocess(self, train_set):
        """
        Add atom id mapping and random BFS order to the training set.
        Compute ``max_edge_unroll`` and ``max_node`` on the training set if not provided.
        """
        remap_atom_type = transforms.RemapAtomType(train_set.atom_types)
        train_set.transform = transforms.Compose([
            train_set.transform,
            remap_atom_type,
            transforms.RandomBFSOrder(),
        ])
        self.id2atom = remap_atom_type.id2atom
        self.atom2id = remap_atom_type.atom2id

        if self.max_edge_unroll is None or self.max_node is None:
            self.max_edge_unroll = 0
            self.max_node = 0

            train_set = tqdm(train_set, "Computing max number of nodes and edge unrolling")
            for sample in train_set:
                graph = sample["graph"]
                if graph.edges.numel():
                    edge_unroll = (graph.edges[0] - graph.edges[1]).abs().max()
                    self.max_edge_unroll = max(self.max_edge_unroll, edge_unroll)
                self.max_node = max(self.max_node, graph.n_node)

            logger.warning("max node = %d, max edge unroll = %d", self.max_node, self.max_edge_unroll)

        self.node_baseline = ops.zeros(self.max_node + 1)
        self.edge_baseline = ops.zeros(self.max_node + 1)

    def loss_fn(self, batch):
        if self.criterion == "nll":
            loss = self.density_estimated_loss(batch)
        elif self.criterion == "ppo":
            loss = self.reinforced_loss(batch)
        else:
            raise ValueError(f"Unknown criterion `{self.criterion}`")
        return loss

    def reinforced_loss(self, graph):
        """_summary_

        Args:
            graph (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        loss = 0
        if self.batch_id % self.agent_update_interval == 0:
            self.agent_node_model.load_ckpt(self.node_model.state_dict())
            self.agent_edge_model.load_ckpt(self.edge_model.state_dict())
        self.batch_id += 1

        graph = self.generate(len(graph), off_policy=True, early_stop=True)
        if graph.n_nodes.max() == 1:
            logger.error("Generation results collapse to singleton molecules")

        reward = ops.zeros(len(graph))
        for task in self.task:
            if task == "plogp":
                plogp = metrics.penalized_logp(graph)
                self.update_best_result(graph, plogp, "Penalized logP")
                reward += (plogp / self.reward_temperature).exp()

                if plogp.max() > 5:
                    print(f"Penalized logP max = {plogp.max()}")
                    print(self.best_results["Penalized logP"])

            elif task == "qed":
                qed = metrics.qed(graph)
                self.update_best_result(graph, qed, "QED")
                reward += (qed / self.reward_temperature).exp()

                if qed.max() > 0.93:
                    print(f"QED max = {qed.max()}")
                    print(self.best_results["QED"])
            else:
                raise ValueError(f"Unknown task `{task}`")

        # There graph-level features will broadcast to all masked graphs
        graph.reward = reward
        graph.ori_n_nodes = graph.n_nodes

        masked_graph, node_target = self.mask_node(graph)
        # reward reshaping
        reward = masked_graph.reward
        masked_graph.atom_type = self.id2atom[masked_graph.atom_type]
        reward = reward * self.gamma ** (masked_graph.ori_n_nodes - masked_graph.n_nodes).astype(ms.float32)

        # per graph size reward baseline
        weight = ops.ones(masked_graph.n_nodes.shape)
        baseline = util.scatter_add(reward, masked_graph.n_nodes, n_axis=self.max_node + 1) / \
            (util.scatter_add(weight, masked_graph.n_nodes, n_axis=self.max_node + 1) + self.eps)
        self.node_baseline = self.node_baseline * self.baseline_momentum + baseline * (1 - self.baseline_momentum)
        reward -= self.node_baseline[masked_graph.n_nodes]
        reward += masked_graph.is_valid
        masked_graph.atom_type = self.atom2id[masked_graph.atom_type]

        log_likelihood = self.node_model(masked_graph, node_target)
        agent_log_likelihood = self.agent_node_model(masked_graph, node_target)
        objective = functional.clipped_policy_gradient_objective(log_likelihood, agent_log_likelihood, reward)
        objective = objective.mean()
        loss += -objective

        masked_graph, edge_target, edge = self.mask_edge(graph)
        # reward reshaping
        reward = masked_graph.reward
        masked_graph.atom_type = self.id2atom[masked_graph.atom_type]
        reward = reward * self.gamma ** (masked_graph.ori_n_nodes - masked_graph.n_nodes).astype(ms.float32)

        # per graph size reward baseline
        weight = ops.oness(masked_graph.n_nodes.shape)
        baseline = util.scatter_add(reward, masked_graph.n_nodes, n_axis=self.max_node + 1) / \
            (util.scatter_add(weight, masked_graph.n_nodes, n_axis=self.max_node + 1) + self.eps)
        self.edge_baseline = self.edge_baseline * self.baseline_momentum + baseline * (1 - self.baseline_momentum)
        reward -= self.edge_baseline[masked_graph.n_nodes]
        reward += masked_graph.is_valid
        masked_graph.atom_type = self.atom2id[masked_graph.atom_type]

        log_likelihood = self.edge_model(masked_graph, edge_target, edge)
        agent_log_likelihood = self.agent_edge_model(masked_graph, edge_target, edge)
        objective = functional.clipped_policy_gradient_objective(log_likelihood, agent_log_likelihood, reward)
        objective = objective.mean()
        loss += -objective

        return loss

    def density_estimated_loss(self, graph):
        """_summary_

        Args:
            graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        loss = 0
        masked_graph, node_target = self.mask_node(graph)
        log_likelihood = self.node_model(masked_graph, node_target, None)
        log_likelihood = log_likelihood.mean()
        loss += -log_likelihood

        masked_graph, edge_target, edge = self.mask_edge(graph)
        log_likelihood = self.edge_model(masked_graph, edge_target, edge)
        log_likelihood = log_likelihood.mean()
        loss += -log_likelihood

        return loss

    def generate(self, n_sample, max_resample=20, off_policy=False, early_stop=False, verbose=0):
        """_summary_

        Args:
            n_sample (_type_): _description_
            max_resample (int, optional): _description_. Defaults to 20.
            off_policy (bool, optional): _description_. Defaults to False.
            early_stop (bool, optional): _description_. Defaults to False.
            verbose (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        n_relation = self.n_bond_type - 1

        if off_policy:
            node_model = self.agent_node_model
            edge_model = self.agent_edge_model
        else:
            node_model = self.node_model
            edge_model = self.edge_model

        edge_list = ops.zeros((1, 3), dtype=ms.int64)
        n_nodes = ops.zeros(n_sample, dtype=ms.int64)
        n_edges = ops.zeros_like(n_nodes)
        atom_type = ops.zeros(0, dtype=ms.int64)
        graph = MoleculeBatch(edge_list, n_nodes=n_nodes, n_edges=n_edges,
                              n_relation=n_relation, atom_type=atom_type, bond_type=edge_list[:, -1])
        completed = ops.zeros(n_sample, dtype=ms.bool_)

        for node_in in range(self.max_node):
            atom_pred = node_model.sample(graph)
            # why we add atom_pred even if it is completed?
            # because we need to batch edge model over (node_in, node_out), even on completed graphs
            atom_type, n_nodes = self._append(atom_type, n_nodes, atom_pred)
            graph = node_graph = MoleculeBatch(edge_list, n_nodes=n_nodes, n_edges=n_edges,
                                               n_relation=n_relation, node_type=atom_type, edge_type=edge_list[:, -1])

            start = max(0, node_in - self.max_edge_unroll)
            for node_out in range(start, node_in):
                is_valid = completed.clone()
                edge = ops.stack([node_in, node_out]).repeat(n_sample, 1)
                bond_pred = (self.n_bond_type - 1) * ops.ones(n_sample, dtype=ms.int64)
                for _ in range(max_resample):
                    # only resample invalid graphs
                    mask = ~is_valid
                    bond_pred[mask] = edge_model.sample(graph, edge)[mask]
                    # check valency
                    mask = (bond_pred < edge_model.input_dim - 1) & ~completed
                    edge_pred = ops.concat([edge, bond_pred.expand_dims(-1)], axis=-1)
                    tmp_edge_list, tmp_n_edges = self._append(edge_list, n_edges, edge_pred, mask)
                    edge_pred = ops.concat([edge.flip(-1), bond_pred.expand_dims(-1)], axis=-1)
                    tmp_edge_list, tmp_n_edges = self._append(tmp_edge_list, tmp_n_edges, edge_pred, mask)
                    tmp_graph = MoleculeBatch(tmp_edge_list, atom_type=self.id2atom[atom_type],
                                              bond_type=tmp_edge_list[:, -1],
                                              n_nodes=n_nodes, n_edges=tmp_n_edges, n_relation=n_relation)

                    is_valid = tmp_graph.is_valid | completed

                    if is_valid.all():
                        break

                if not is_valid.all() and verbose:
                    n_invalid = n_sample - is_valid.sum().item()
                    n_working = n_sample - completed.sum().item()
                    logger.warning("edge (%d, %d): %d / %d molecules are invalid even after %d resampling",
                                   node_in, node_out, n_invalid, n_working, max_resample)

                mask = (bond_pred < edge_model.input_dim - 1) & ~completed
                edge_pred = ops.concat([edge, bond_pred.expand_dims(-1)], axis=-1)
                edge_list, n_edges = self._append(edge_list, n_edges, edge_pred, mask)
                edge_pred = ops.concat([edge.flip(-1), bond_pred.expand_dims(-1)], axis=-1)
                edge_list, n_edges = self._append(edge_list, n_edges, edge_pred, mask)
                graph = MoleculeBatch(edge_list, bond_type=edge_list[:, -1], n_nodes=n_nodes, n_edges=n_edges,
                                      n_relation=n_relation, atom_type=atom_type)

            if node_in > 0:
                assert (graph.n_edges[completed] == node_graph.n_edges[completed]).all()
                completed |= graph.n_edges == node_graph.n_edges
                if early_stop:
                    graph.atom_type = self.id2atom[graph.atom_type]
                    completed |= ~graph.is_valid()
                    graph.atom_type = self.atom2id[graph.atom_type]
                if completed.all():
                    break

        # remove isolated atoms
        index = graph.degree_out() > 0
        # keep at least the first atom for each graph
        index[graph.cum_nodes - graph.n_nodes] = 1
        graph = graph.subgraph(index)
        graph.atom_type = self.id2atom[graph.atom_type]

        graph = graph[graph.is_valid_rdkit]
        return graph

    def mask_node(self, graph, metric=None):
        if self.n_node_sample == -1:
            masked_graph, node_target = self.all_node(graph)
            if metric is not None:
                metric["node mask / graph"] = ms.Tensor([len(masked_graph) / len(graph)])
        else:
            masked_graph, node_target = self.sample_node(graph, self.n_node_sample)
        return masked_graph, node_target

    def mask_edge(self, graph, metric=None):
        if self.n_edge_sample == -1:
            masked_graph, edge_target, edge = self.all_edge(graph)
            if metric is not None:
                metric["edge mask / graph"] = ms.Tensor([len(masked_graph) / len(graph)])
        else:
            masked_graph, edge_target, edge = self.sample_edge(graph, self.n_edge_sample)
        return masked_graph, edge_target, edge

    def sample_node(self, graph, n_sample):
        """_summary_

        Args:
            graph (_type_): _description_
            n_sample (_type_): _description_

        Returns:
            _type_: _description_
        """
        graph = graph.repeat(n_sample)
        n_nodes = graph.n_nodes
        n_keep_nodes = ops.randint(0, n_nodes, len(graph))  # [0, n_nodes)

        starts = graph.cum_nodes - graph.n_nodes
        ends = starts + n_keep_nodes
        mask = functional.multi_slice_mask(starts, ends, graph.n_node)

        new_graph = graph.subgraph(mask)
        target = graph.subgraph(ends).atom_type
        return new_graph, target

    def all_node(self, graph):
        starts, ends, valid = self._all_prefix_slice(graph.n_nodes)

        n_repeat = len(starts) // len(graph)
        graph = graph.repeat(n_repeat)
        mask = functional.multi_slice_mask(starts, ends, graph.n_node)
        new_graph = graph.subgraph(mask)
        target = graph.subgraph(ends).atom_type

        return new_graph[valid], target[valid]

    def sample_edge(self, graph, n_sample):
        """_summary_

        Args:
            graph (_type_): _description_
            n_sample (_type_): _description_

        Returns:
            _type_: _description_
        """
        if (graph.n_nodes < 2).any():
            graph = graph[graph.n_nodes >= 2]
            warnings.warn("Graphs with less than 2 nodes can't be used for edge generation learning. Dropped")

        lengths = self._valid_edge_prefix_lengths(graph)
        graph = graph.repeat(n_sample)

        n_max_node = graph.n_nodes.max().item()
        n_node2n_dense_edge = ops.arange(n_max_node + 1) ** 2
        n_node2length_idx = ops.sum(lengths.expand_dims(-1) < n_node2n_dense_edge.expand_dims(0), dim=0)
        # uniformly sample a mask from each graph's valid masks
        length_indexes = ops.randint(0, n_node2length_idx[graph.n_nodes], len(graph))
        n_keep_dense_edges = lengths[length_indexes]

        # undirected: all upper triangular edge ids are flipped to lower triangular ids
        # 1 -> 2, 4 -> 6, 5 -> 7
        node_index = graph.edges - graph.offsets
        node_in, node_out = node_index
        node_large = node_index.max(axis=-1)[0]
        node_small = node_index.min(axis=-1)[0]
        edge_id = node_large ** 2 + (node_in >= node_out) * node_large + node_small
        undirected_edge_id = node_large * (node_large + 1) + node_small

        edge_mask = undirected_edge_id < n_keep_dense_edges[graph.edge2graph]
        circum_box_size = (n_keep_dense_edges + 1.0).sqrt().ceil().astype(ms.int64)
        starts = graph.cum_nodes - graph.n_nodes
        ends = starts + circum_box_size
        node_mask = functional.multi_slice_mask(starts, ends, graph.n_node)
        # compact nodes so that succeeding nodes won't affect graph pooling
        new_graph = graph.edge_mask(edge_mask).node_mask(node_mask, compact=True)

        positive_edge = edge_id == n_keep_dense_edges[graph.edge2graph]
        positive_graph = util.scatter_add(positive_edge.astype(
            ms.int64), graph.edge2graph, axis=0, n_axis=len(graph)).astype(ms.bool_)
        target = (self.n_bond_type - 1) * ops.ones(graph.batch_size, dtype=ms.int64)
        target[positive_graph] = graph.edge_list[positive_edge, 2]

        node_in = circum_box_size - 1
        node_out = n_keep_dense_edges - node_in * circum_box_size
        edge = ops.stack([node_in, node_out], axis=-1)

        return new_graph, target, edge

    def all_edge(self, graph):
        """_summary_

        Args:
            graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        if (graph.n_nodes < 2).any():
            graph = graph[graph.n_nodes >= 2]
            warnings.warn("Graphs with less than 2 nodes can't be used for edge generation learning. Dropped")

        lengths = self._valid_edge_prefix_lengths(graph)

        starts, ends, valid = self._all_prefix_slice(graph.n_nodes ** 2, lengths)

        n_keep_dense_edges = ends - starts
        n_repeat = len(starts) // len(graph)
        graph = graph.repeat(n_repeat)

        # undirected: all upper triangular edge ids are flipped to lower triangular ids
        # 1 -> 2, 4 -> 6, 5 -> 7
        node_index = graph.edge_list - graph.offsets
        node_in, node_out = node_index.t()
        node_large = node_index.max(axis=-1)[0]
        node_small = node_index.min(axis=-1)[0]
        edge_id = node_large ** 2 + (node_in >= node_out) * node_large + node_small
        undirected_edge_id = node_large * (node_large + 1) + node_small

        edge_mask = undirected_edge_id < n_keep_dense_edges[graph.edge2graph]
        circum_box_size = (n_keep_dense_edges + 1.0).sqrt().ceil().astype(ms.int64)
        starts = graph.cum_nodes - graph.n_nodes
        ends = starts + circum_box_size
        node_mask = functional.multi_slice_mask(starts, ends, graph.n_node)
        # compact nodes so that succeeding nodes won't affect graph pooling
        new_graph = graph.edge_mask(edge_mask).node_mask(node_mask, compact=True)

        positive_edge = edge_id == n_keep_dense_edges[graph.edge2graph]
        positive_graph = util.scatter_add(positive_edge.astype(ms.int64), graph.edge2graph,
                                          axis=0, n_axis=len(graph)).astype(ms.bool_)
        target = (self.n_bond_type - 1) * ops.ones(graph.batch_size, dtype=ms.int64)
        target[positive_graph] = graph.edge_list[:, positive_edge]

        node_in = circum_box_size - 1
        node_out = n_keep_dense_edges - node_in * circum_box_size
        edge = ops.stack([node_in, node_out], axis=-1)

        return new_graph[valid], target[valid], edge[valid]

    def update_best_result(self, graph, score, task):
        best_results = self.best_results[task]
        for s, i in zip(*score.sort(descending=True)):
            if len(best_results) == self.top_k and s < best_results[-1][0]:
                break
            best_results.append((s, graph[i].to_smiles()))
            best_results.sort(reverse=True)
            best_results = best_results[:self.top_k]
        self.best_results[task] = best_results

    def _all_prefix_slice(self, n_xs, lengths=None):
        """_summary_

        Args:
            n_xs (_type_): _description_
            lengths (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # extract a bunch of slices that correspond to the following n_repeat * n masks
        # ------ repeat 0 -----
        # The shape of graphs[0]: [0, 0, ..., 0]
        # ...
        # The shape of graphs[-1]: [0, 0, ..., 0]
        # ------ repeat 1 -----
        # The shape of graphs[0]: [1, 0, ..., 0]
        # ...
        # The shape of graphs[-1]: [1, 0, ..., 0]
        # ...
        # ------ repeat -1 -----
        # The shape of graphs[0]: [1, ..., 1, 0]
        # ...
        # The shape of graphs[-1]: [1, ..., 1, 0]
        cum_xs = n_xs.cumsum(0)
        starts = cum_xs - n_xs
        if lengths is None:
            n_max_x = n_xs.max().item()
            lengths = ops.arange(n_max_x)

        pack_offsets = ops.arange(len(lengths)) * cum_xs[-1]
        # starts, lengths, ends: (n_repeat, n_graph)
        starts = starts.expand_dims(0) + pack_offsets.expand_dims(-1)
        valid = lengths.expand_dims(-1) <= n_xs.expand_dims(0) - 1
        lengths = ops.min(lengths.expand_dims(-1), n_xs.expand_dims(0) - 1)
        ends = starts + lengths

        starts = starts.flatten()
        ends = ends.flatten()
        valid = valid.flatten()

        return starts, ends, valid

    def _valid_edge_prefix_lengths(self, graph):
        """_summary_

        Args:
            graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        # valid prefix lengths are across a batch, according to the largest graph
        n_max_node = graph.n_nodes.max().item()
        # edge id in an adjacency (snake pattern)
        #    in
        # o 0 1 4
        # u 2 3 5
        # t 6 7 8
        lengths = ops.arange(n_max_node ** 2)
        circum_box_size = (lengths + 1.0).sqrt().ceil().astype(ms.int64)
        # only keep lengths that ends in the lower triangular part of adjacency matrix
        lengths = lengths[lengths >= circum_box_size * (circum_box_size - 1)]
        # lengths looks like: [0, 2, 3, 6, 7, 8, ...]
        # n_node2length_idx looks like: [0, 1, 4, 6, ...]
        # n_edge_unrolls
        # 0
        # 1 0
        # 2 1 0
        n_edge_unrolls = (lengths + 1.0).sqrt().ceil().astype(ms.int64) ** 2 - lengths - 1
        # The shape of n_edge_unrolls look like: [0, 1, 0, 2, 1, 0, ...]
        # remove lengths that unroll too much. they always lead to empty targets
        lengths = lengths[(n_edge_unrolls <= self.max_edge_unroll) & (n_edge_unrolls > 0)]

        return lengths


class GCPNGeneration(core.Cell):
    """
    The graph generative model from `Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation`_.
    .. _Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation:
        https://papers.nips.cc/paper/7877-graph-convolutional-policy-network-for-goal-directed-molecular-graph-generation.pdf
    Parameters:
        model (nn.Module): graph representation model
        atom_types (list or set): set of all possible atom types
        task (str or list of str, optional): property optimization task(s)
        max_edge_unroll (int, optional): max node id difference.
            If not provided, use the statistics from the training set.
        max_node (int, optional): max number of node.
            If not provided, use the statistics from the training set.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``nll`` and ``ppo``.
        agent_update_interval (int, optional): update the agent every n batch
        gamma (float, optional): reward discount rate
        reward_temperature (float, optional): temperature for reward. Higher temperature encourages larger mean reward,
            while lower temperature encourages larger maximal reward.
        baseline_momentum (float, optional): momentum for value function baseline
    """

    eps = 1e-10
    top_k = 10
    _option_members = {"task", "criterion"}

    def __init__(self, model, atom_types, max_edge_unroll=None, max_node=None, task=(), criterion="nll",
                 hidden_dim_mlp=128, agent_update_interval=10, gamma=0.9, reward_temperature=1, baseline_momentum=0.9):
        super().__init__()
        self.model = model
        self.task = task
        self.max_edge_unroll = max_edge_unroll
        self.max_node = max_node
        self.criterion = criterion
        self.hidden_dim_mlp = hidden_dim_mlp
        self.agent_update_interval = agent_update_interval
        self.gamma = gamma
        self.reward_temperature = reward_temperature
        self.baseline_momentum = baseline_momentum
        self.best_results = defaultdict(list)
        self.batch_id = 0

        remap_atom_type = transforms.RemapAtomType(atom_types)
        self.id2atom = remap_atom_type.id2atom
        self.atom2id = remap_atom_type.atom2id

        self.new_atom_embeddings = ms.Parameter(ops.zeros(self.id2atom.size(0), self.model.output_dim))
        nn.init.normal_(self.new_atom_embeddings, mean=0, std=0.1)
        self.inp_dim_stop = self.model.output_dim
        self.mlp_stop = layers.MultiLayerPerceptron(self.inp_dim_stop, [self.hidden_dim_mlp, 2], activation='tanh')

        self.inp_dim_node1 = self.model.output_dim + self.model.output_dim
        self.mlp_node1 = layers.MultiLayerPerceptron(self.inp_dim_node1, [self.hidden_dim_mlp, 1], activation='tanh')
        self.inp_dim_node2 = 2 * self.model.output_dim + self.model.output_dim
        self.mlp_node2 = layers.MultiLayerPerceptron(self.inp_dim_node2, [self.hidden_dim_mlp, 1], activation='tanh')
        self.inp_dim_edge = 2 * self.model.output_dim
        self.mlp_edge = layers.MultiLayerPerceptron(
            self.inp_dim_edge, [self.hidden_dim_mlp, self.model.n_relation], activation='tanh')

        self.agent_model = copy.deepcopy(self.model)
        self.agent_new_atom_embeddings = copy.deepcopy(self.new_atom_embeddings)
        self.agent_mlp_stop = copy.deepcopy(self.mlp_stop)
        self.agent_mlp_node1 = copy.deepcopy(self.mlp_node1)
        self.agent_mlp_node2 = copy.deepcopy(self.mlp_node2)
        self.agent_mlp_edge = copy.deepcopy(self.mlp_edge)
        self.moving_baseline = None

    def preprocess(self, train_set):
        """
        Add atom id mapping and random BFS order to the training set.

        Compute ``max_edge_unroll`` and ``max_node`` on the training set if not provided.
        """
        train_set.transform = transforms.Compose([
            train_set.transform,
            transforms.RandomBFSOrder(),
        ])

        if self.max_edge_unroll is None or self.max_node is None:
            self.max_edge_unroll = 0
            self.max_node = 0

            train_set = tqdm(train_set, "Computing max number of nodes and edge unrolling")
            for sample in train_set:
                graph = sample["graph"]
                if graph.edge_list.numel():
                    edge_unroll = (graph.edge_list[:, 0] - graph.edge_list[:, 1]).abs().max().item()
                    self.max_edge_unroll = max(self.max_edge_unroll, edge_unroll)
                self.max_node = max(self.max_node, graph.n_node)

            logger.warning("max node = %d, max edge unroll = %d", self.max_node, self.max_edge_unroll)

        self.moving_baseline = ops.zeros(self.max_node + 1)

    def loss_fn(self, batch):
        """_summary_

        Args:
            batch (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        loss = 0
        for criterion, weight in self.criterion.items():
            if criterion == "nll":
                loss_ = self.mle_forward(batch)
                loss += loss_ * weight
            elif criterion == "ppo":
                loss_ = self.reinforce_forward(batch)
                loss += loss_ * weight
            else:
                raise ValueError(f"Unknown criterion `{criterion}`")

        return loss

    def predict(self, graph, label_dict, use_agent=False):
        """_summary_

        Args:
            graph (_type_): _description_
            label_dict (_type_): _description_
            use_agent (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # step1: get node/graph embeddings
        if not use_agent:
            output = self.model(graph, graph.node_feat.astype(ms.float32))
        else:
            output = self.agent_model(graph, graph.node_feat.astype(ms.float32))

        # The shape is (n_graph * 16)
        extended_node2graph = ops.arange(graph.n_nodes.size(0),
                                         ).expand_dims(1).repeat([1, self.id2atom.size(0)]).view(-1)
        extended_node2graph = ops.concat((graph.node2graph, extended_node2graph))  # (n_node + 16 * n_graph)

        graph_feat_per_node = output["graph_feat"][extended_node2graph]

        # step2: predict stop
        stop_feat = output["graph_feat"]  # (n_graph, n_out)
        if not use_agent:
            stop_logits = self.mlp_stop(stop_feat)  # (n_graph, 2)
        else:
            stop_logits = self.agent_mlp_stop(stop_feat)  # (n_graph, 2)

        if label_dict is None:
            return stop_logits
        # step3: predict first node: node1
        node1_feat = output["node_feat"]  # (n_node, n_out)

        # The shape of node1_feat is (n_node + 16 * n_graph, n_out)
        node1_feat = ops.concat((node1_feat,
                                 self.new_atom_embeddings.repeat([graph.n_nodes.size(0), 1])), 0)

        node2_feat_node2 = node1_feat.clone()  # (n_node + 16 * n_graph, n_out)
        # cat graph emb
        node1_feat = ops.concat((node1_feat, graph_feat_per_node), 1)

        if not use_agent:
            node1_logits = self.mlp_node1(node1_feat).squeeze(1)  # (n_node + 16 * n_graph)
        else:
            node1_logits = self.agent_mlp_node1(node1_feat).squeeze(1)  # (n_node + 16 * n_graph)

        # mask the extended part
        mask = ops.zeros(node1_logits.size())
        mask[:graph.n_node] = 1
        node1_logits = ops.where(mask > 0, node1_logits, -10000.0*ops.ones(node1_logits.size()))

        # step4: predict second node: node2

        node1_index_per_graph = (graph.cum_nodes - graph.n_nodes) + label_dict["label1"]  # (n_graph)
        node1_index = node1_index_per_graph[extended_node2graph]  # (n_node + 16 * n_graph)
        node2_feat_node1 = node1_feat[node1_index]  # (n_node + 16 * n_graph, n_out)
        node2_feat = ops.concat((node2_feat_node1, node2_feat_node2), 1)  # (n_node + 16 * n_graph, 2n_out)
        if not use_agent:
            node2_logits = self.mlp_node2(node2_feat).squeeze(1)  # (n_node + 16 * n_graph)
        else:
            node2_logits = self.agent_mlp_node2(node2_feat).squeeze(1)  # (n_node + 16 * n_graph)

        # mask the selected node1
        mask = ops.zeros(node2_logits.size())
        mask[node1_index_per_graph] = 1
        node2_logits = ops.where(mask == 0, node2_logits, -10000.0 *
                                 ops.ones(node2_logits.size()))

        # step5: predict edge type
        # if an entry is non-negative, this is a new added node. (n_graph)
        is_new_node = label_dict["label2"] - graph.n_nodes
        graph_offset = ops.arange(graph.n_nodes.size(0))
        node2_index_per_graph = ops.where(is_new_node >= 0,
                                          graph.n_node + graph_offset * self.id2atom.size(0) + is_new_node,
                                          label_dict["label2"] + graph.cum_nodes - graph.n_nodes)  # (n_graph)

        edge_feat_node1 = node2_feat_node2[node1_index_per_graph]  # (n_graph, n_out)
        edge_feat_node2 = node2_feat_node2[node2_index_per_graph]  # (n_graph, n_out)
        edge_feat = ops.concat((edge_feat_node1, edge_feat_node2), 1)  # (n_graph, 2n_out)
        if not use_agent:
            edge_logits = self.mlp_edge(edge_feat)  # (n_graph, n_relation)
        else:
            edge_logits = self.agent_mlp_edge(edge_feat)  # (n_graph, n_relation)

        index_dict = {
            "node1_index_per_graph": node1_index_per_graph,
            "node2_index_per_graph": node2_index_per_graph,
            "extended_node2graph": extended_node2graph
        }
        return stop_logits, node1_logits, node2_logits, edge_logits, index_dict

    def reinforce_forward(self, batch):
        """_summary_

        Args:
            batch (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        all_loss = 0
        metric = {}

        # generation takes less time when early_stop=True
        graph = self.generate(len(batch["graph"]), max_resample=20, off_policy=True, max_step=40 * 2)
        if (not graph) or graph.n_nodes.max() == 1:
            logger.error("Generation results collapse to singleton molecules")

            return all_loss

        reward = ops.zeros(len(graph))
        for task in self.task:
            if task == "plogp":
                plogp = metrics.penalized_logp(graph)
                metric["Penalized logP"] = plogp.mean()
                metric["Penalized logP (max)"] = plogp.max()
                self.update_best_result(graph, plogp, "Penalized logP")
                # TODO:
                reward += (plogp / self.reward_temperature).exp()

                if plogp.max().item() > 5:
                    print(f"Penalized logP max = {plogp.max()}")
                    print(self.best_results["Penalized logP"])

            elif task == "qed":
                qed = metrics.qed(graph)
                metric["QED"] = qed.mean()
                metric["QED (max)"] = qed.max()
                self.update_best_result(graph, qed, "QED")

                reward += (qed / self.reward_temperature).exp()

                if qed.max().item() > 0.93:
                    print(f"QED max = {qed.max()}")
                    print(self.best_results["QED"])
            else:
                raise ValueError(f"Unknown task `{task}`")

        # these graph-level features will broadcast to all masked graphs
        with graph.graph():
            graph.reward = reward
            graph.original_n_nodes = graph.n_nodes

        is_training = self.training
        # easily got nan if BN is trained
        self.bn_eval()

        stop_graph, stop_label1, stop_label2, stop_label3, stop_label4 = self.all_stop(graph)
        edge_graph, edge_label1, edge_label2, edge_label3, edge_label4 = self.all_edge(graph)

        graph = self._cat([stop_graph, edge_graph])
        label1_target = ops.concat([stop_label1, edge_label1])
        label2_target = ops.concat([stop_label2, edge_label2])
        label3_target = ops.concat([stop_label3, edge_label3])
        label4_target = ops.concat([stop_label4, edge_label4])
        label_dict = {"label1": label1_target, "label2": label2_target,
                      "label3": label3_target, "label4": label4_target}

        # reward reshaping
        reward = graph.reward
        reward = reward * self.gamma ** (graph.original_n_nodes - graph.n_nodes).astype(ms.float32)

        # per graph size reward baseline
        weight = ops.ones(graph.n_nodes.shape)
        baseline = util.scatter_add(reward, graph.n_nodes, n_axis=self.max_node + 1) / \
            (util.scatter_add(weight, graph.n_nodes, n_axis=self.max_node + 1) + self.eps)
        # TODO:
        self.moving_baseline = self.moving_baseline * self.baseline_momentum + baseline * (1 - self.baseline_momentum)
        reward -= self.moving_baseline[graph.n_nodes]
        reward += graph.is_valid

        # calculate object
        stop_logits, node1_logits, node2_logits, edge_logits, index_dict = self.predict(graph, label_dict)
        old_stop_logits, old_node1_logits, old_node2_logits, old_edge_logits, old_index_dict = self.predict(
            graph, label_dict, use_agent=True)

        stop_prob = ops.log_softmax(stop_logits, axis=-1)
        node1_prob = util.scatter_log_softmax(node1_logits, index_dict["extended_node2graph"])
        node2_prob = util.scatter_log_softmax(node2_logits, index_dict["extended_node2graph"])
        edge_prob = ops.log_softmax(edge_logits, axis=-1)
        old_stop_prob = ops.log_softmax(old_stop_logits, axis=-1)
        old_node1_prob = util.scatter_log_softmax(old_node1_logits, old_index_dict["extended_node2graph"])
        old_node2_prob = util.scatter_log_softmax(old_node2_logits, old_index_dict["extended_node2graph"])
        old_edge_prob = ops.log_softmax(old_edge_logits, axis=-1)

        cur_logp = stop_prob[:, 0] + node1_prob[index_dict["node1_index_per_graph"]] \
            + node2_prob[index_dict["node2_index_per_graph"]] + \
            ops.gather(edge_prob, label3_target.view(-1, 1), -1).view(-1)
        cur_logp[label4_target == 1] = stop_prob[:, 1][label4_target == 1]

        old_logp = old_stop_prob[:, 0] + old_node1_prob[old_index_dict["node1_index_per_graph"]] \
            + old_node2_prob[index_dict["node2_index_per_graph"]] + \
            ops.gather(old_edge_prob, label3_target.view(-1, 1), -1).view(-1)
        old_logp[label4_target == 1] = old_stop_prob[:, 1][label4_target == 1]
        objective = functional.clipped_policy_gradient_objective(cur_logp, old_logp, reward)
        objective = objective.mean()
        metric["PPO objective"] = objective
        all_loss += (-objective)

        self.bn_train(is_training)

        return all_loss, metric

    def mle_forward(self, graph):
        """_summary_

        Args:
            graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        all_loss = 0
        metric = {}

        stop_graph, stop_label1, stop_label2, stop_label3, stop_label4 = self.all_stop(graph)
        edge_graph, edge_label1, edge_label2, edge_label3, edge_label4 = self.all_edge(graph)

        graph = self._cat([stop_graph, edge_graph])
        label1_target = ops.concat([stop_label1, edge_label1])
        label2_target = ops.concat([stop_label2, edge_label2])
        label3_target = ops.concat([stop_label3, edge_label3])
        label4_target = ops.concat([stop_label4, edge_label4])
        label_dict = {"label1": label1_target, "label2": label2_target,
                      "label3": label3_target, "label4": label4_target}
        stop_logits, node1_logits, node2_logits, edge_logits, index_dict = self.predict(graph, label_dict)

        loss_stop = ops.nll_loss(ops.log_softmax(stop_logits, axis=-1), label4_target, reduction='none')
        loss_stop = 0.5 * (ops.mean(loss_stop[label4_target == 0]) + ops.mean(loss_stop[label4_target == 1]))

        metric["stop bce loss"] = loss_stop
        all_loss += loss_stop

        loss_node1 = -util.scatter_log_softmax(node1_logits, index_dict["extended_node2graph"])
        loss_node1 = loss_node1[index_dict["node1_index_per_graph"]]
        loss_node1 = ops.mean(loss_node1[label4_target == 0])
        metric["node1 loss"] = loss_node1
        all_loss += loss_node1

        loss_node2 = -util.scatter_log_softmax(node2_logits, index_dict["extended_node2graph"])
        loss_node2 = loss_node2[index_dict["node2_index_per_graph"]]
        loss_node2 = ops.mean(loss_node2[label4_target == 0])
        metric["node2 loss"] = loss_node2
        all_loss += loss_node2

        loss_edge = ops.nll_loss(ops.log_softmax(edge_logits, axis=-1), label3_target, reduction='none')

        loss_edge = ops.mean(loss_edge[label4_target == 0])
        metric["edge loss"] = loss_edge
        all_loss += loss_edge

        metric["total loss"] = all_loss

        pred = stop_logits, node1_logits, node2_logits, edge_logits
        target = label1_target, label2_target, label3_target, label4_target, index_dict

        metric.update(self.evaluate(pred, target))

        return all_loss, metric

    def evaluate(self, pred, target):
        """_summary_

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        stop_logits, node1_logits, node2_logits, edge_logits = pred
        _, _, label3_target, label4_target, index_dict = target
        metric = {}
        stop_acc = ops.argmax(stop_logits, -1) == label4_target
        metric["stop acc"] = stop_acc.astype(ms.float32).mean()

        node1_pred = util.scatter_max(node1_logits, index_dict["extended_node2graph"])[1]
        node1_acc = node1_pred == index_dict["node1_index_per_graph"]
        metric["node1 acc"] = node1_acc[label4_target == 0].astype(ms.float32).mean()

        node2_pred = util.scatter_max(node2_logits, index_dict["extended_node2graph"])[1]
        node2_acc = node2_pred == index_dict["node2_index_per_graph"]
        metric["node2 acc"] = node2_acc[label4_target == 0].astype(ms.float32).mean()

        edge_acc = ops.argmax(edge_logits, -1) == label3_target
        metric["edge acc"] = edge_acc[label4_target == 0].astype(ms.float32).mean()
        return metric

    def bn_train(self, mode=True):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train(mode)

    def bn_eval(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

    def update_best_result(self, graph, score, task):
        """_summary_

        Args:
            graph (_type_): _description_
            score (_type_): _description_
            task (_type_): _description_
        """
        score = score.cpu()
        best_results = self.best_results[task]
        for s, i in zip(*score.sort(descending=True)):
            s = s.item()
            i = i.item()
            if len(best_results) == self.top_k and s < best_results[-1][0]:
                break
            best_results.append((s, graph[i].to_smiles()))
            best_results.sort(reverse=True)
            best_results = best_results[:self.top_k]
        self.best_results[task] = best_results

    def generate(self, n_sample, max_resample=20, off_policy=False, max_step=30 * 2, initial_smiles="C"):
        """_summary_

        Args:
            n_sample (_type_): _description_
            max_resample (int, optional): _description_. Defaults to 20.
            off_policy (bool, optional): _description_. Defaults to False.
            max_step (_type_, optional): _description_. Defaults to 30*2.
            initial_smiles (str, optional): _description_. Defaults to "C".

        Returns:
            _type_: _description_
        """
        is_training = self.training
        self.eval()

        graph = Molecule.from_smiles(initial_smiles, kekulized=True, node_feat="symbol").repeat(n_sample)

        result = []
        for i in range(max_step):
            new_graph = self._apply_action(graph, off_policy, max_resample, verbose=1)
            if i == max_step - 1:
                # last step, collect all graph that is valid
                result.append(new_graph[(new_graph.n_nodes <= (self.max_node))])
            else:
                result.append(new_graph[new_graph.is_stopped | (new_graph.n_nodes == (self.max_node))])

                is_continue = (~new_graph.is_stopped) & (new_graph.n_nodes < (self.max_node))
                graph = new_graph[is_continue]
                if not graph:
                    break

        self.train(is_training)

        result = self._cat(result)
        return result

    def all_stop(self, graph):
        if (graph.n_nodes < 2).any():
            graph = graph[graph.n_nodes >= 2]
            warnings.warn("Graphs with less than 2 nodes can't be used for stop prediction learning. Dropped")

        label1 = ops.zeros(len(graph), dtype=ops.int64)
        label2 = ops.zeros_like(label1)
        label3 = ops.zeros_like(label1)
        return graph, label1, label2, label3, ops.ones(len(graph), dtype=ops.int64)

    def all_edge(self, graph):
        """_summary_

        Args:
            graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        if (graph.n_nodes < 2).any():
            graph = graph[graph.n_nodes >= 2]
            warnings.warn("Graphs with less than 2 nodes can't be used for edge generation learning. Dropped")

        lengths = self._valid_edge_prefix_lengths(graph)

        starts, ends, _ = self._all_prefix_slice(graph.n_nodes ** 2, lengths)

        n_keep_dense_edges = ends - starts
        n_repeat = len(starts) // len(graph)
        graph = graph.repeat(n_repeat)

        # undirected: all upper triangular edge ids are flipped to lower triangular ids
        # 1 -> 2, 4 -> 6, 5 -> 7
        node_index = graph.edge_list - graph.offsets
        node_in, node_out = node_index.t()
        node_large = node_index.max(axis=-1)[0]
        node_small = node_index.min(axis=-1)[0]
        edge_id = node_large ** 2 + (node_in >= node_out) * node_large + node_small
        undirected_edge_id = node_large * (node_large + 1) + node_small

        edge_mask = undirected_edge_id < n_keep_dense_edges[graph.edge2graph]
        circum_box_size = (n_keep_dense_edges + 1.0).sqrt().ceil().astype(ms.int64)

        # check whether we need to add a new node for the current edge
        masked_undirected_edge_id = ops.where(edge_mask, undirected_edge_id, -ops.ones(undirected_edge_id.size(),
                                                                                       dtype=ops.int64))
        current_circum_box_size = util.scatter_max(masked_undirected_edge_id, graph.edge2graph, axis=0)[0]
        current_circum_box_size = (current_circum_box_size + 1.0).sqrt().ceil().astype(ms.int64)
        is_new_node_edge = (circum_box_size > current_circum_box_size).astype(ms.int64)

        starts = graph.cum_nodes - graph.n_nodes
        ends = starts + circum_box_size - is_new_node_edge
        node_mask = functional.multi_slice_mask(starts, ends, graph.n_node)
        # compact nodes so that succeeding nodes won't affect graph pooling
        new_graph = graph.edge_mask(edge_mask).node_mask(node_mask, compact=True)

        positive_edge = edge_id == n_keep_dense_edges[graph.edge2graph]
        positive_graph = util.scatter_add(positive_edge.astype(
            ms.int64), graph.edge2graph, axis=0, n_axis=len(graph)).astype(ms.bool_)

        target = (self.model.n_relation) * ops.ones(graph.batch_size, dtype=ops.long)
        target[positive_graph] = graph.edge_list[:, 2][positive_edge]

        node_in = circum_box_size - 1
        node_out = n_keep_dense_edges - node_in * circum_box_size
        # if we need to add a new node, what will be its atomid?
        new_node_atomid = self.atom2id[graph.atom_type[starts + node_in]]

        # keep only the positive graph, as we will add an edge at each step
        new_graph = new_graph[positive_graph]
        target = target[positive_graph]
        node_in = node_in[positive_graph]
        node_out = node_out[positive_graph]
        is_new_node_edge = is_new_node_edge[positive_graph]
        new_node_atomid = new_node_atomid[positive_graph]

        node_in_extend = new_graph.n_nodes + new_node_atomid
        node_in_final = ops.where(is_new_node_edge == 0, node_in, node_in_extend)

        return new_graph, node_out, node_in_final, target, ops.zeros(node_out.shape)

    # generation step
    # 1. top-1 action
    # 2. apply action
    def _construct_dist(self, prob_, graph):
        """_summary_

        Args:
            prob_ (_type_): _description_
            graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_size = max(graph.n_nodes) + self.id2atom.size(0)
        probs = ops.zeros((len(graph), max_size))
        start = (graph.cum_nodes - graph.n_nodes)[graph.node2graph]
        start = ops.arange(graph.n_node) - start
        index = ops.arange(graph.n_nodes.size(0)) * max_size
        index = index[graph.node2graph] + start
        probs[index] = prob_[:graph.n_node]

        start_extend = ops.arange(len(self.id2atom)).repeat(
            graph.n_nodes.size())  # (n_graph * 16)
        index_extend = ops.arange(len(graph)) * max_size + graph.n_nodes
        index2graph = ops.arange(len(graph))
        index_extend = index_extend[index2graph] + start_extend
        probs[index_extend] = prob_[graph.n_node:]
        probs = probs.view(len(graph.n_nodes), max_size)
        return Categorical(probs), probs  # (n_graph, max_size)

    def _sample_action(self, graph, off_policy):
        """_summary_

        Args:
            graph (_type_): _description_
            off_policy (_type_): _description_

        Returns:
            _type_: _description_
        """
        if off_policy:
            model = self.agent_model
            new_atom_embeddings = self.agent_new_atom_embeddings
            mlp_stop = self.agent_mlp_stop
            mlp_node1 = self.agent_mlp_node1
            mlp_node2 = self.agent_mlp_node2
            mlp_edge = self.agent_mlp_edge
        else:
            model = self.model
            new_atom_embeddings = self.new_atom_embeddings
            mlp_stop = self.mlp_stop
            mlp_node1 = self.mlp_node1
            mlp_node2 = self.mlp_node2
            mlp_edge = self.mlp_edge

        # step1: get feature
        output = model(graph, graph.node_feat.astype(ms.float32))

        extended_node2graph = ops.arange(len(graph)).repeat(
            len(self.id2atom))  # (n_graph * 16)
        extended_node2graph = ops.concat((graph.node2graph, extended_node2graph))  # (n_node + 16 * n_graph)

        graph_feat_per_node = output["graph_feat"][extended_node2graph]

        # step2: predict stop
        stop_feat = output["graph_feat"]  # (n_graph, n_out)
        stop_logits = mlp_stop(stop_feat)  # (n_graph, 2)
        stop_prob = ops.softmax(stop_logits, -1)  # (n_graph, 2)
        stop_prob_dist = Categorical(stop_prob)
        stop_pred = stop_prob_dist.sample()
        # step3: predict first node: node1

        node1_feat = output["node_feat"]  # (n_node, n_out)
        # The shape is (n_node + 16 * n_graph, n_out)
        node1_feat = ops.concat((node1_feat,
                                 new_atom_embeddings.repeat([graph.n_nodes.size(0), 1])), 0)
        node2_feat_node2 = node1_feat.clone()  # (n_node + 16 * n_graph, n_out)

        node1_feat = ops.concat((node1_feat, graph_feat_per_node), 1)

        node1_logits = mlp_node1(node1_feat).squeeze(1)  # (n_node + 16 * n_graph)
        # mask the extended part
        mask = ops.zeros(node1_logits.size())
        mask[:graph.n_node] = 1
        node1_logits = ops.where(mask > 0, node1_logits, -10000.0*ops.ones(node1_logits.size()))

        node1_prob = util.scatter_softmax(node1_logits, extended_node2graph)  # (n_node + 16 * n_graph)
        node1_prob_dist, _ = self._construct_dist(node1_prob, graph)  # (n_graph, max)

        node1_pred = node1_prob_dist.sample()  # (n_graph)
        node1_index_per_graph = node1_pred + (graph.cum_nodes - graph.n_nodes)
        # step4: predict second node: node2
        node1_index = node1_index_per_graph[extended_node2graph]  # (n_node + 16 * n_graph)
        node2_feat_node1 = node1_feat[node1_index]  # (n_node + 16 * n_graph, n_out)

        node2_feat = ops.concat((node2_feat_node1, node2_feat_node2), 1)  # (n_node + 16 * n_graph, 2n_out)
        node2_logits = mlp_node2(node2_feat).squeeze(1)  # (n_node + 16 * n_graph)

        # mask the selected node1
        mask = ops.zeros(node2_logits.size())
        mask[node1_index_per_graph] = 1
        node2_logits = ops.where(mask == 0, node2_logits, -10000.0 *
                                 ops.ones(node2_logits.size()))
        node2_prob = util.scatter_softmax(node2_logits, extended_node2graph)  # (n_node + 16 * n_graph)
        node2_prob_dist, _ = self._construct_dist(node2_prob, graph)  # (n_graph, max)
        node2_pred = node2_prob_dist.sample()  # (n_graph,)
        is_new_node = node2_pred - graph.n_nodes
        graph_offset = ops.arange(graph.n_nodes.size(0))
        node2_index_per_graph = ops.where(is_new_node >= 0,
                                          graph.n_node + graph_offset * self.id2atom.size(0) + is_new_node,
                                          node2_pred + graph.cum_nodes - graph.n_nodes)

        # step5: predict edge type
        edge_feat_node1 = node2_feat_node2[node1_index_per_graph]  # (n_graph, n_out)
        edge_feat_node2 = node2_feat_node2[node2_index_per_graph]  # (n_graph, n_out)
        edge_feat = ops.concat((edge_feat_node1, edge_feat_node2), 1)  # (n_graph, 2n_out)
        edge_logits = mlp_edge(edge_feat)
        edge_prob = ops.softmax(edge_logits, -1)  # (n_graph, 3)
        edge_prob_dist = Categorical(edge_prob)
        edge_pred = edge_prob_dist.sample()

        return stop_pred, node1_pred, node2_pred, edge_pred

    def _top1_action(self, graph, off_policy):
        """_summary_

        Args:
            graph (_type_): _description_
            off_policy (_type_): _description_

        Returns:
            _type_: _description_
        """
        if off_policy:
            model = self.agent_model
            new_atom_embeddings = self.agent_new_atom_embeddings
            mlp_stop = self.agent_mlp_stop
            mlp_node1 = self.agent_mlp_node1
            mlp_node2 = self.agent_mlp_node2
            mlp_edge = self.agent_mlp_edge
        else:
            model = self.model
            new_atom_embeddings = self.new_atom_embeddings
            mlp_stop = self.mlp_stop
            mlp_node1 = self.mlp_node1
            mlp_node2 = self.mlp_node2
            mlp_edge = self.mlp_edge

        # step1: get feature
        output = model(graph, graph.node_feat.astype(ms.float32))
        # The shape is (n_graph * 16)
        extended_node2graph = ops.arange(graph.n_nodes.size(0)
                                         ).expand_dims(1).repeat([1, self.id2atom.size(0)]).view(-1)
        extended_node2graph = ops.concat((graph.node2graph, extended_node2graph))  # (n_node + 16 * n_graph)

        graph_feat_per_node = output["graph_feat"][extended_node2graph]

        # step2: predict stop
        stop_feat = output["graph_feat"]  # (n_graph, n_out)
        stop_logits = mlp_stop(stop_feat)  # (n_graph, 2)
        stop_pred = ops.argmax(stop_logits, -1)  # (n_graph,)
        # step3: predict first node: node1

        node1_feat = output["node_feat"]  # (n_node, n_out)
        # The shape of node1_feat is (n_node + 16 * n_graph, n_out)
        node1_feat = ops.concat((node1_feat,
                                 new_atom_embeddings.repeat([graph.n_nodes.size(0), 1])), 0)
        node2_feat_node2 = node1_feat.clone()  # (n_node + 16 * n_graph, n_out)

        node1_feat = ops.concat((node1_feat, graph_feat_per_node), 1)

        node1_logits = mlp_node1(node1_feat).squeeze(1)  # (n_node + 16 * n_graph)
        # mask the extended part
        mask = ops.zeros(node1_logits.size())
        mask[:graph.n_node] = 1
        node1_logits = ops.where(mask > 0, node1_logits, -10000.0*ops.ones(node1_logits.size()))

        node1_index_per_graph = util.scatter_max(node1_logits, extended_node2graph)[1]  # (n_node + 16 * n_graph)
        node1_pred = node1_index_per_graph - (graph.cum_nodes - graph.n_nodes)

        # step4: predict second node: node2
        node1_index = node1_index_per_graph[extended_node2graph]  # (n_node + 16 * n_graph)
        node2_feat_node1 = node1_feat[node1_index]  # (n_node + 16 * n_graph, n_out)

        node2_feat = ops.concat((node2_feat_node1, node2_feat_node2), 1)  # (n_node + 16 * n_graph, 2n_out
        node2_logits = mlp_node2(node2_feat).squeeze(1)  # (n_node + 16 * n_graph)

        # mask the selected node1
        mask = ops.zeros(node2_logits.size())
        mask[node1_index_per_graph] = 1
        node2_logits = ops.where(mask == 0, node2_logits, -10000.0 *
                                 ops.ones(node2_logits.size()))
        node2_index_per_graph = util.scatter_max(node2_logits, extended_node2graph)[1]  # (n_node + 16 * n_graph)

        is_new_node = node2_index_per_graph - graph.n_node  # non negative if is new node
        graph_offset = ops.arange(graph.n_nodes.size(0))
        node2_pred = ops.where(is_new_node >= 0, graph.n_nodes + is_new_node - graph_offset * self.id2atom.size(0),
                               node2_index_per_graph - (graph.cum_nodes - graph.n_nodes))

        # step5: predict edge type
        edge_feat_node1 = node2_feat_node2[node1_index_per_graph]  # (n_graph, n_out)
        edge_feat_node2 = node2_feat_node2[node2_index_per_graph]  # (n_graph, n_out)
        edge_feat = ops.concat((edge_feat_node1, edge_feat_node2), 1)  # (n_graph, 2n_out)
        edge_logits = mlp_edge(edge_feat)
        edge_pred = ops.argmax(edge_logits, -1)

        return stop_pred, node1_pred, node2_pred, edge_pred

    def _apply_action(self, graph, off_policy, max_resample=10, verbose=0):
        """_summary_

        Args:
            graph (_type_): _description_
            off_policy (_type_): _description_
            max_resample (int, optional): _description_. Defaults to 10.
            verbose (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        # stopped graph is removed, initialize is_valid as False
        is_valid = ops.zeros(len(graph), dtype=ms.bool_)
        stop_action = ops.zeros(len(graph), dtype=ops.int64)
        node1_action = ops.zeros(len(graph), dtype=ops.int64)
        node2_action = ops.zeros(len(graph), dtype=ops.int64)
        edge_action = ops.zeros(len(graph), dtype=ops.int64)

        for _ in range(max_resample):
            # maximal resample time
            mask = ~is_valid
            if max_resample == 1:
                tmp_stop_action, tmp_node1_action, tmp_node2_action, tmp_edge_action = \
                    self._top1_action(graph, off_policy)
            else:
                tmp_stop_action, tmp_node1_action, tmp_node2_action, tmp_edge_action = \
                    self._sample_action(graph, off_policy)

            stop_action[mask] = tmp_stop_action[mask]
            node1_action[mask] = tmp_node1_action[mask]
            node2_action[mask] = tmp_node2_action[mask]
            edge_action[mask] = tmp_edge_action[mask]

            stop_action[graph.n_nodes <= 5] = 0
            # tmp add new nodes
            has_new_node = (node2_action >= graph.n_nodes) & (stop_action == 0)
            new_atom_id = (node2_action - graph.n_nodes)[has_new_node]
            new_atom_type = self.id2atom[new_atom_id]

            atom_type, n_nodes = functional.extend(graph.atom_type, graph.n_nodes, new_atom_type, has_new_node)

            # tmp cast to regular node ids
            node2_action = ops.where(has_new_node, graph.n_nodes, node2_action)

            # tmp modify edges
            new_edge = ops.stack([node1_action, node2_action], axis=-1)
            edge_list = graph.edge_list.clone()
            bond_type = graph.bond_type.clone()
            edge_list -= graph.offsets
            is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(axis=-1) & \
                (stop_action[graph.edge2graph] == 0)
            has_modified_edge = util.scatter_max(is_modified_edge.astype(
                ms.int64), graph.edge2graph, n_axis=len(graph))[0] > 0
            bond_type[is_modified_edge] = edge_action[has_modified_edge]
            edge_list[is_modified_edge, 2] = edge_action[has_modified_edge]
            # tmp modify reverse edges
            new_edge = new_edge.flip(-1)
            is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(axis=-1) & \
                (stop_action[graph.edge2graph] == 0)
            bond_type[is_modified_edge] = edge_action[has_modified_edge]
            edge_list[is_modified_edge, 2] = edge_action[has_modified_edge]

            # tmp add new edges
            has_new_edge = (~has_modified_edge) & (stop_action == 0)
            new_edge_list = ops.stack([node1_action, node2_action, edge_action], axis=-1)[has_new_edge]
            bond_type = functional.extend(bond_type, graph.n_edges, edge_action[has_new_edge], has_new_edge)[0]
            edge_list, n_edges = functional.extend(edge_list, graph.n_edges, new_edge_list, has_new_edge)

            # tmp add reverse edges
            new_edge_list = ops.stack([node2_action, node1_action, edge_action], axis=-1)[has_new_edge]
            bond_type = functional.extend(bond_type, n_edges, edge_action[has_new_edge], has_new_edge)[0]
            edge_list, n_edges = functional.extend(edge_list, n_edges, new_edge_list, has_new_edge)

            tmp_graph = type(graph)(edge_list, atom_type=atom_type, bond_type=bond_type, n_nodes=n_nodes,
                                    n_edges=n_edges, n_relation=graph.n_relation)
            is_valid = tmp_graph.is_valid | (stop_action == 1)
            if is_valid.all():
                break
        if not is_valid.all() and verbose:
            n_invalid = len(graph) - is_valid.sum().item()
            n_working = len(graph)
            logger.warning("%d / %d molecules are invalid even after %d resampling",
                           n_invalid, n_working, max_resample)

        # apply the true action
        # inherit attributes
        data_dict = graph.data_dict
        meta_dict = graph.meta_dict
        for key in ["atom_type", "bond_type"]:
            data_dict.pop(key)
        # pad 0 for node / edge attributes
        for k, v in data_dict.items():
            if "node" in meta_dict[k]:
                shape = (len(new_atom_type), *v.shape[1:])
                new_data = ops.zeros(shape, dtype=v.dtype)
                data_dict[k] = functional.extend(v, graph.n_nodes, new_data, has_new_node)[0]
            if "edge" in meta_dict[k]:
                shape = (len(new_edge_list) * 2, *v.shape[1:])
                new_data = ops.zeros(shape, dtype=v.dtype)
                data_dict[k] = functional.extend(v, graph.n_edges, new_data, has_new_edge * 2)[0]

        new_graph = type(graph)(edge_list, atom_type=atom_type, bond_type=bond_type, n_nodes=n_nodes,
                                n_edges=n_edges, n_relation=graph.n_relation,
                                meta_dict=meta_dict, **data_dict)
        with new_graph.graph():
            new_graph.is_stopped = stop_action == 1

        new_graph, feat_valid = self._update_molecule_feat(new_graph)

        return new_graph[feat_valid]

    def _update_molecule_feat(self, graphs):
        """_summary_

        Args:
            graphs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # This function is very slow
        mols = graphs.to_molecule(ignore_error=True)
        valid = [mol is not None for mol in mols]
        valid = ops.tensor(valid)
        new_graphs = type(graphs).from_molecule(mols, kekulize=True, atom_feat="symbol")

        node_feat = ops.zeros(graphs.n_node, *new_graphs.node_feat.shape[1:],
                              dtype=new_graphs.node_feat.dtype)
        edge_feat = ops.zeros(graphs.n_edge, *new_graphs.edge_feat.shape[1:],
                              dtype=new_graphs.edge_feat.dtype)
        bond_type = ops.zeros(graphs.bond_type.shape)
        node_mask = valid[graphs.node2graph]
        edge_mask = valid[graphs.edge2graph]
        node_feat[node_mask] = new_graphs.node_feat
        edge_feat[edge_mask] = new_graphs.edge_feat
        bond_type[edge_mask] = new_graphs.bond_type

        with graphs.node():
            graphs.node_feat = node_feat
        with graphs.edge():
            graphs.edge_feat = edge_feat
            graphs.bond_type = bond_type

        return graphs, valid

    def _all_prefix_slice(self, n_xs, lengths=None):
        """_summary_

        Args:
            n_xs (_type_): _description_
            lengths (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # extract a bunch of slices that correspond to the following n_repeat * n masks
        # ------ repeat 0 -----
        # graphs[0] looks like: [0, 0, ..., 0]
        # ...
        # graphs[-1] looks like: [0, 0, ..., 0]
        # ------ repeat 1 -----
        # graphs[0] looks like: [1, 0, ..., 0]
        # ...
        # graphs[-1] looks like: [1, 0, ..., 0]
        # ...
        # ------ repeat -1 -----
        # graphs[0] looks like: [1, ..., 1, 0]
        # ...
        # graphs[-1] looks like: [1, ..., 1, 0]
        cum_xs = n_xs.cumsum(0)
        starts = cum_xs - n_xs
        if lengths is None:
            n_max_x = n_xs.max().item()
            lengths = ops.arange(n_max_x)

        pack_offsets = ops.arange(len(lengths)) * cum_xs[-1]
        # starts, lengths, ends: (n_repeat, n_graph)
        starts = starts.expand_dims(0) + pack_offsets.expand_dims(-1)
        valid = lengths.expand_dims(-1) <= n_xs.expand_dims(0) - 1
        lengths = ops.min(lengths.expand_dims(-1), n_xs.expand_dims(0) - 1).clamp(0)
        ends = starts + lengths

        starts = starts.flatten()
        ends = ends.flatten()
        valid = valid.flatten()

        return starts, ends, valid

    def _valid_edge_prefix_lengths(self, graph):
        """_summary_

        Args:
            graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_max_node = graph.n_nodes.max().item()
        # edge id in an adjacency (snake pattern)
        #    in
        # o 0 1 4
        # u 2 3 5
        # t 6 7 8
        lengths = ops.arange(n_max_node ** 2)
        circum_box_size = (lengths + 1.0).sqrt().ceil().astype(ms.int64)
        # only keep lengths that ends in the lower triangular part of adjacency matrix
        lengths = lengths[lengths >= circum_box_size * (circum_box_size - 1)]
        # lengths looks like [0, 2, 3, 6, 7, 8, ...]
        # n_node2length_idx loooks like [0, 1, 4, 6, ...]
        # n_edge_unrolls
        # 0
        # 1 0
        # 2 1 0
        n_edge_unrolls = (lengths + 1.0).sqrt().ceil().astype(ms.int64) ** 2 - lengths - 1
        # n_edge_unrolls looks like: [0, 1, 0, 2, 1, 0, ...]
        # remove lengths that unroll too much. they always lead to empty targets.
        lengths = lengths[(n_edge_unrolls <= self.max_edge_unroll) & (n_edge_unrolls > 0)]

        return lengths

    def _cat(self, graphs):
        """_summary_

        Args:
            graphs (_type_): _description_

        Returns:
            _type_: _description_
        """
        for i, graph in enumerate(graphs):
            if not isinstance(graph, GraphBatch):
                graphs[i] = graph.pack([graph])

        edge_list = ops.concat([graph.edge_list for graph in graphs])
        pack_n_nodes = ops.stack([graph.n_node for graph in graphs])
        pack_n_edges = ops.stack([graph.n_edge for graph in graphs])
        pack_cum_edges = pack_n_edges.cumsum(0)
        graph_index = pack_cum_edges < len(edge_list)
        pack_offsets = util.scatter_add(pack_n_nodes[graph_index], pack_cum_edges[graph_index],
                                        n_axis=len(edge_list))
        pack_offsets = pack_offsets.cumsum(0)

        edge_list[:, :2] += pack_offsets.expand_dims(-1)
        offsets = ops.concat([graph.offsets for graph in graphs]) + pack_offsets

        edge_weight = ops.concat([graph.edge_weight for graph in graphs])
        n_nodes = ops.concat([graph.n_nodes for graph in graphs])
        n_edges = ops.concat([graph.n_edges for graph in graphs])
        n_relation = graphs[0].n_relation
        assert all(graph.n_relation == n_relation for graph in graphs)

        # only keep attributes that exist in all graphs
        keys = set(graphs[0].meta_dict.keys())
        for graph in graphs:
            keys = keys.intersection(graph.meta_dict.keys())

        meta_dict = {k: graphs[0].meta_dict[k] for k in keys}
        data_dict = {}
        for k in keys:
            data_dict[k] = ops.concat([graph.data_dict[k] for graph in graphs])

        return type(graphs[0])(edge_list, edge_weight=edge_weight,
                               n_nodes=n_nodes, n_edges=n_edges, n_relation=n_relation, offsets=offsets,
                               meta_dict=meta_dict, **data_dict)

    def _append(self, data, n_xs, inputs, mask=None):
        """_summary_

        Args:
            data (_type_): _description_
            n_xs (_type_): _description_
            inputs (_type_): _description_
            mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if mask is None:
            mask = ops.ones_like(n_xs, dtype=ops.bool_)
        new_n_xs = n_xs + mask
        new_cum_xs = new_n_xs.cumsum(0)
        new_n_x = new_cum_xs[-1].item()
        new_data = ops.zeros(new_n_x, *data.shape[1:], dtype=data.dtype)
        starts = new_cum_xs - new_n_xs
        ends = starts + n_xs
        index = functional.multi_slice_mask(starts, ends, new_n_x)
        new_data[index] = data
        new_data[~index] = inputs[mask]
        return new_data, new_n_xs
