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
retrosynthesis
"""
import sys
import inspect
import logging

import mindspore as ms
from mindspore import ops
from mindspore import nn

from .. import layers
from .. import metrics
from ..data import GraphBatch, MoleculeBatch
from .. import core
from .. import util
from ..util import functional
from ..core import Registry as R
from .. import transforms


logger = logging.getLogger(__name__)


@R.register("scenario.CenterIdentification")
class CenterIdentification(nn.Cell):
    """
    Reaction center identification task.

    This class is a part of retrosynthesis prediction.

    Parameters:
        model (nn.Module): graph representation model
        feature (str or list of str, optional): additional features for prediction. Available features are
            reaction: type of the reaction
            graph: graph representation of the product
            atom: original atom feature
            bond: original bond feature
        n_mlp_layer (int, optional): number of MLP layers
    """

    _option_members = {"feature"}

    def __init__(self, model, feature=("reaction", "graph", "atom", "bond"), n_mlp_layer=2):
        super().__init__()
        self.model = model
        self.n_mlp_layer = n_mlp_layer
        self.feature = feature
        self.n_reaction = None
        self.n_relation = None
        self.edge_mlp = None
        self.node_mlp = None

    def preprocess(self, train_set):
        """_summary_

        Args:
            train_set (_type_): _description_

        Raises:
            ValueError: _description_
        """
        reaction_types = set()
        bond_types = set()
        for sample in train_set:
            reaction_types.add(sample["reaction"])
            for graph in sample["graph"]:
                bond_types.update(graph.edge_list[:, 2].tolist())
        self.n_reaction = len(reaction_types)
        self.n_relation = len(bond_types)
        node_feature_dim = train_set[0]["graph"][0].node_feature.shape[-1]
        edge_feature_dim = train_set[0]["graph"][0].edge_feature.shape[-1]

        node_dim = self.model.output_dim
        edge_dim = 0
        graph_dim = 0
        for feature_ in sorted(self.feature):
            if feature_ == "reaction":
                graph_dim += self.n_reaction
            elif feature_ == "graph":
                graph_dim += self.model.output_dim
            elif feature_ == "atom":
                node_dim += node_feature_dim
            elif feature_ == "bond":
                edge_dim += edge_feature_dim
            else:
                raise ValueError(f"Unknown feature `{feature_}`")

        node_dim += graph_dim  # inherit graph features
        edge_dim += node_dim * 2  # inherit node features

        hidden_dims = [self.model.output_dim] * (self.n_mlp_layer - 1)
        self.edge_mlp = layers.MLP(edge_dim, hidden_dims + [1])
        self.node_mlp = layers.MLP(node_dim, hidden_dims + [1])

    def construct(self, *batch):
        """_summary_

        Returns:
            _type_: _description_
        """
        all_loss = 0
        metric = {}
        batch, _ = core.args_from_dict(batch)
        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)
        metric.update(self.evaluate(pred, target))

        target, size = target
        target = functional.variadic_max(target, size)[1]
        loss = functional.variadic_cross_entropy(pred, target, size)

        all_loss += loss

        return all_loss, metric

    def target(self, batch):
        """_summary_

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        _, product = batch["graph"]
        graph = product.directed()

        target = self._collate(graph.edge_label, graph.node_label, graph)
        size = graph.n_edges + graph.n_nodes
        return target, size

    def predict(self, batch, all_loss=None, metric=None):
        """_summary_

        Args:
            batch (_type_): _description_
            all_loss (_type_, optional): _description_. Defaults to None.
            metric (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        _, product = batch["graph"]
        output = self.model(product, product.node_feature.astype(ms.float32), all_loss, metric)

        graph = product.directed()

        node_feature = [output["node_feature"]]
        edge_feature = []
        graph_feature = []
        for feature_ in sorted(self.feature):
            if feature_ == "reaction":
                reaction_feature = ops.zeros(len(graph), self.n_reaction)
                reaction_feature.scatter_(1, batch["reaction"].expand_dims(-1), 1)
                graph_feature.append(reaction_feature)
            elif feature_ == "graph":
                graph_feature.append(output["graph_feature"])
            elif feature_ == "atom":
                node_feature.append(graph.node_feature.astype(ms.float32))
            elif feature_ == "bond":
                edge_feature.append(graph.edge_feature.astype(ms.float32))
            else:
                raise ValueError(f"Unknown feature `{feature_}`")

        graph_feature = ops.concat(graph_feature, axis=-1)
        # inherit graph features
        node_feature.append(graph_feature[graph.node2graph])
        node_feature = ops.concat(node_feature, axis=-1)
        # inherit node features
        edge_feature.append(node_feature[graph.edge_list[:, :2]].flatten(1))
        edge_feature = ops.concat(edge_feature, axis=-1)

        edge_pred = self.edge_mlp(edge_feature).squeeze(-1)
        node_pred = self.node_mlp(node_feature).squeeze(-1)

        pred = self._collate(edge_pred, node_pred, graph)

        return pred

    def predict_synthon(self, batch, k=1):
        """
        Predict top-k synthons from target molecules.

        Parameters:
            batch (dict): batch of target molecules
            k (int, optional): return top-k results

        Returns:
            list of dict: top k records.
                Each record is a batch dict of keys ``synthon``, ``n_synthon``, ``reaction_center``,
                ``log_likelihood`` and ``reaction``.
        """
        pred = self.predict(batch)
        _, size = self.target(batch)
        logp = functional.variadic_log_softmax(pred, size)

        _, product = batch["graph"]
        graph = product.directed()
        with graph.graph():
            graph.product_id = ops.arange(len(graph))

        graph = graph.repeat(k)
        reaction = batch["reaction"].repeat(k)
        with graph.graph():
            graph.split_id = ops.arange(k).repeat(len(graph) // k)

        logp, center_topk = functional.variadic_topk(logp, size, k)
        logp = logp.flatten()
        center_topk = center_topk.flatten()

        is_edge = center_topk < graph.n_edges
        node_index = center_topk + graph.cum_nodes - graph.n_nodes - graph.n_edges
        edge_index = center_topk + graph.cum_edges - graph.n_edges
        center_topk_shifted = ops.concat([-ops.ones(1, dtype=ms.int64),
                                          center_topk[:-1]])
        product_id_shifted = ops.concat([-ops.ones(1, dtype=ms.int64),
                                         graph.product_id[:-1]])
        is_duplicate = (center_topk == center_topk_shifted) & (graph.product_id == product_id_shifted)
        node_index = node_index[~is_edge]
        edge_index = edge_index[is_edge]
        edge_mask = ~functional.as_mask(edge_index, graph.n_edge)

        reaction_center = ops.zeros((len(graph), 2), dtype=ms.int64)
        reaction_center[is_edge] = graph.atom_map[graph.edge_list[edge_index, :2]]
        reaction_center[~is_edge, 0] = graph.atom_map[node_index]

        # remove the edges from products
        graph = graph.edge_mask(edge_mask)
        graph = graph[~is_duplicate]
        reaction_center = reaction_center[~is_duplicate]
        logp = logp[~is_duplicate]
        reaction = reaction[~is_duplicate]
        synthon, n_synthon = graph.connected_components()
        synthon = synthon.undirected()  # (< n_graph * k)

        result = {
            "synthon": synthon,
            "n_synthon": n_synthon,
            "reaction_center": reaction_center,
            "log_likelihood": logp,
            "reaction": reaction,
        }

        return result

    def _collate(self, edge_data, node_data, graph):
        """_summary_

        Args:
            edge_data (_type_): _description_
            node_data (_type_): _description_
            graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_data = ops.zeros(len(edge_data) + len(node_data), *edge_data.shape[1:],
                             dtype=edge_data.dtype)
        cum_xs = graph.cum_edges + graph.cum_nodes
        n_xs = graph.n_edges + graph.n_nodes
        starts = cum_xs - n_xs
        ends = starts + graph.n_edges
        index = functional.multi_slice_mask(starts, ends, cum_xs[-1])
        new_data[index] = edge_data
        new_data[~index] = node_data
        return new_data


@R.register("scenario.Retrosynthesis")
class Retrosynthesis(nn.Cell):
    """
    Retrosynthesis task.

    This class wraps pretrained center identification and synthon completion modeules into a pipeline.

    Parameters:
        center_identification (CenterIdentification): sub task of center identification
        synthon_completion (SynthonCompletion): sub task of synthon completion
        center_topk (int, optional): number of reaction centers to predict for each product
        n_synthon_beam (int, optional): size of beam search for each synthon
        max_prediction (int, optional): max number of final predictions for each product
        metric (str or list of str, optional): metric(s). Available metrics are ``top-K``.
    """

    _option_members = {"metric"}

    def __init__(self, center_identification, synthon_completion, center_topk=2, n_synthon_beam=10, max_prediction=20,
                 metric=("top-1", "top-3", "top-5", "top-10")):
        super().__init__()
        self.center_identification = center_identification
        self.synthon_completion = synthon_completion
        self.center_topk = center_topk
        self.n_synthon_beam = n_synthon_beam
        self.max_prediction = max_prediction
        self.metric = metric

    def load_state_dict(self, state_dict, strict=True):
        if not strict:
            raise ValueError("Retrosynthesis only supports load_state_dict() with strict=True")
        keys = set(state_dict.keys())
        for model in [self.center_identification, self.synthon_completion]:
            if set(model.state_dict().keys()) == keys:
                return model.load_state_dict(state_dict, strict)
        raise RuntimeError("Neither of sub modules matches with state_dict")

    def predict(self, batch):
        """_summary_

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        synthon_batch = self.center_identification.predict_synthon(batch, self.center_topk)

        synthon = synthon_batch["synthon"]
        n_synthon = synthon_batch["n_synthon"]
        assert (n_synthon >= 1).all() and (n_synthon <= 2).all()
        synthon2split = ops.repeat(n_synthon)
        with synthon.graph():
            synthon.reaction_center = synthon_batch["reaction_center"][synthon2split]
            synthon.split_logp = synthon_batch["log_likelihood"][synthon2split]

        reactant = self.synthon_completion.predict_reactant(synthon_batch, self.n_synthon_beam, self.max_prediction)

        logps = []
        reactant_ids = []
        product_ids = []

        # case 1: one synthon
        is_single = n_synthon[synthon2split[reactant.synthon_id]] == 1
        reactant_id = is_single.nonzero().squeeze(-1)
        logps.append(reactant.split_logp[reactant_id] + reactant.logp[reactant_id])
        product_ids.append(reactant.product_id[reactant_id])
        # pad -1
        reactant_ids.append(ops.stack([reactant_id, -ops.ones(reactant_id.shape)], axis=-1))

        # case 2: two synthons
        # use proposal to avoid O(n^2) complexity
        reactant1 = ops.arange(len(reactant))
        reactant1 = reactant1.expand_dims(-1).expand(-1, self.max_prediction * 2)
        reactant2 = reactant1 + ops.arange(self.max_prediction * 2)
        valid = reactant2 < len(reactant)
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        synthon1 = reactant.synthon_id[reactant1]
        synthon2 = reactant.synthon_id[reactant2]
        valid = (synthon1 < synthon2) & (synthon2split[synthon1] == synthon2split[synthon2])
        reactant1 = reactant1[valid]
        reactant2 = reactant2[valid]
        logps.append(reactant.split_logp[reactant1] + reactant.logp[reactant1] + reactant.logp[reactant2])
        product_ids.append(reactant.product_id[reactant1])
        reactant_ids.append(ops.stack([reactant1, reactant2], axis=-1))

        # combine case 1 & 2
        logps = ops.concat(logps)
        reactant_ids = ops.concat(reactant_ids)
        product_ids = ops.concat(product_ids)

        order = product_ids.argsort()
        logps = logps[order]
        reactant_ids = reactant_ids[order]
        n_prediction = product_ids.bincount()
        logps, topk = functional.variadic_topk(logps, n_prediction, self.max_prediction)
        topk_index = topk + (n_prediction.cumsum(0) - n_prediction).expand_dims(-1)
        topk_index_shifted = ops.concat([-ops.ones((len(topk_index), 1), dtype=ms.int64),
                                         topk_index[:, :-1]], axis=-1)
        is_duplicate = topk_index == topk_index_shifted
        reactant_id = reactant_ids[topk_index]  # (n_graph, k, 2)

        # why we need to repeat the graph?
        # because reactant_id may be duplicated, which is not directly supported by graph indexing
        is_padding = reactant_id == -1
        n_synthon = (~is_padding).sum(axis=-1)
        n_synthon = n_synthon[~is_duplicate]
        logps = logps[~is_duplicate]
        offset = ops.arange(self.max_prediction) * len(reactant)
        reactant_id = reactant_id + offset.view(1, -1, 1)
        reactant_id = reactant_id[~(is_padding | is_duplicate.expand_dims(-1))]
        reactant = reactant.repeat(self.max_prediction)
        reactant = reactant[reactant_id]
        assert n_synthon.sum() == len(reactant)
        synthon2graph = ops.repeat(n_synthon)
        first_synthon = n_synthon.cumsum(0) - n_synthon
        # inherit graph attributes from the first synthon
        data_dict = reactant.data_mask(graph_index=first_synthon, include="graph")[0]
        # merge synthon pairs from the same split into a single graph
        reactant = reactant.merge(synthon2graph)
        with reactant.graph():
            for k, v in data_dict.items():
                setattr(reactant, k, v)
            reactant.logps = logps

        n_prediction = reactant.product_id.bincount()

        return reactant, n_prediction  # (n_graph * k)

    def target(self, batch):
        reactant, _ = batch["graph"]
        reactant = reactant.ion_to_molecule()
        return reactant

    def evaluate(self, preds, targets):
        """_summary_

        Args:
            preds (_type_): _description_
            targets (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        preds, n_prediction = preds
        infinity = sys.maxint

        metric = {}
        ranking = []
        # any better solution for parallel graph isomorphism?
        cum_prediction = n_prediction.cumsum(0)
        for i, target in enumerate(targets):
            target_smiles = target.to_smiles(isomeric=False, atom_map=False, canonical=True)
            offset = cum_prediction[i] - n_prediction[i]
            for j in range(n_prediction[i]):
                pred_smiles = preds[offset + j].to_smiles(isomeric=False, atom_map=False, canonical=True)
                if pred_smiles == target_smiles:
                    break
            else:
                j = infinity
            ranking.append(j + 1)

        ranking = ms.Tensor(ranking)
        for metric_ in self.metric:
            if metric_.startswith("top-"):
                threshold = int(metric_[4:])
                score = (ranking <= threshold).astype(ms.float32).mean()
                metric["top-%d accuracy" % threshold] = score
            else:
                raise ValueError("Unknown metric `%s`" % metric_)

        return metric


@R.register("scenario.SynthonCompletion")
class SynthonCompletion(nn.Cell):
    """
    Synthon completion task.

    This class is a part of retrosynthesis prediction.

    Parameters:
        model (nn.Module): graph representation model
        feature (str or list of str, optional): additional features for prediction. Available features are
            reaction: type of the reaction
            graph: graph representation of the synthon
            atom: original atom feature
        n_mlp_layer (int, optional): number of MLP layers
    """

    _option_members = {"feature"}

    def __init__(self, model, feature=("reaction", "graph", "atom"), n_mlp_layer=2):
        super().__init__()
        self.model = model
        self.n_mlp_layer = n_mlp_layer
        self.feature = feature
        self.input_linear = nn.Dense(2, self.model.input_dim)
        self.new_atom_feature = None
        self.node_in_mlp = None
        self.node_out_mlp = None
        self.edge_mlp = None
        self.bond_mlp = None
        self.stop_mlp = None
        self.feature_kwargs = None
        self.n_reaction = None
        self.n_atom_type = None
        self.n_bond_type = None

    def preprocess(self, train_set):
        """_summary_

        Args:
            train_set (_type_): _description_

        Raises:
            ValueError: _description_
        """
        reaction_types = set()
        atom_types = set()
        bond_types = set()
        for sample in train_set:
            reaction_types.add(sample["reaction"])
            for graph in sample["graph"]:
                atom_types.update(graph.atom_type.tolist())
                bond_types.update(graph.edge_list[:, 2].tolist())
        # TODO: only for fast debugging, to remove
        atom_types = ms.Tensor(sorted(atom_types))
        atom2id = -ops.ones(atom_types.max() + 1, dtype=ms.int64)
        atom2id[atom_types] = ops.arange(len(atom_types))
        setattr(self, "id2atom", atom_types)
        setattr(self, "atom2id", atom2id)
        self.n_reaction = len(reaction_types)
        self.n_atom_type = len(atom_types)
        self.n_bond_type = len(bond_types)
        node_feature_dim = train_set[0]["graph"][0].node_feature.shape[-1]

        dataset = train_set
        dataset.transform = transforms.Compose([
            dataset.transform,
            transforms.RandomBFSOrder(),
        ])
        sig = inspect.signature(MoleculeBatch.from_molecule)
        keys = set(sig.parameters.keys())
        kwargs = dataset.config_dict()
        feature_kwargs = {}
        for k, v in kwargs.items():
            if k in keys:
                feature_kwargs[k] = v
        self.feature_kwargs = feature_kwargs

        node_dim = self.model.output_dim
        edge_dim = 0
        graph_dim = 0
        for feature_ in sorted(self.feature):
            if feature_ == "reaction":
                graph_dim += self.n_reaction
            elif feature_ == "graph":
                graph_dim += self.model.output_dim
            elif feature_ == "atom":
                node_dim += node_feature_dim
            else:
                raise ValueError(f"Unknown feature `{feature_}`")

        self.new_atom_feature = nn.Embedding(self.n_atom_type, node_dim)

        node_dim += graph_dim  # inherit graph features
        edge_dim += node_dim * 2  # inherit node features

        hidden_dims = [self.model.output_dim] * (self.n_mlp_layer - 1)
        self.node_in_mlp = layers.MLP(node_dim, hidden_dims + [1])
        self.node_out_mlp = layers.MLP(edge_dim, hidden_dims + [1])
        self.edge_mlp = layers.MLP(edge_dim, hidden_dims + [1])
        self.bond_mlp = layers.MLP(edge_dim, hidden_dims + [self.n_bond_type])
        self.stop_mlp = layers.MLP(graph_dim, hidden_dims + [1])

    def all_edge(self, reactant, synthon):
        """_summary_

        Args:
            reactant (_type_): _description_
            synthon (_type_): _description_

        Returns:
            _type_: _description_
        """
        graph = reactant.clone()
        node_r2s, edge_r2s, is_new_node, is_new_edge, is_modified_edge, is_reaction_center = \
            self._get_reaction_feature(reactant, synthon)
        with graph.node():
            graph.node_r2s = node_r2s
            graph.is_new_node = is_new_node
            graph.is_reaction_center = is_reaction_center
        with graph.edge():
            graph.edge_r2s = edge_r2s
            graph.is_new_edge = is_new_edge
            graph.is_modified_edge = is_modified_edge

        starts, ends, valid = self._all_prefix_slice(reactant.n_edges)
        n_repeat = len(starts) // len(reactant)
        graph = graph.repeat(n_repeat)

        # autoregressive condition range for each sample
        condition_mask = functional.multi_slice_mask(starts, ends, graph.n_edge)
        # special case: end == graph.n_edge. In this case, valid is always false
        assert ends.max() <= graph.n_edge
        ends = ends.clamp(0, graph.n_edge - 1)
        node_in, node_out, bond_target = graph.edge_list[ends].t()
        # modified edges which don't appear in conditions should keep their old bond types
        # i.e. bond types in synthons
        unmodified = ~condition_mask & graph.is_modified_edge
        unmodified = unmodified.nonzero().squeeze(-1)
        assert not (graph.bond_type[unmodified] == synthon.bond_type[graph.edge_r2s[unmodified]]).any()
        graph.edge_list[unmodified, 2] = synthon.edge_list[graph.edge_r2s[unmodified], 2]

        reverse_target = graph.edge_list[ends][:, [1, 0, 2]]
        is_reverse_target = (graph.edge_list == reverse_target[graph.edge2graph]).all(axis=-1)
        # keep edges that exist in the synthon
        # remove the reverse of new target edges
        edge_mask = (condition_mask & ~is_reverse_target) | ~graph.is_new_edge

        atom_out = graph.atom_type[node_out]
        # keep one supervision for undirected edges
        # remove samples that try to predict existing edges
        valid &= (node_in < node_out) & (graph.is_new_edge[ends] | graph.is_modified_edge[ends])
        graph = graph.edge_mask(edge_mask)

        # sanitize the molecules
        # this will change atom index, so we manually remap the target nodes
        compact_mapping = -ops.ones(graph.n_node, dtype=ms.int64)
        node_mask = graph.degree_in + graph.degree_out > 0
        # special case: for graphs without any edge, the first node should be kept
        index = ops.arange(graph.n_node)
        single_node_mask = (graph.n_edges == 0)[graph.node2graph] & \
                           (index == (graph.cum_nodes - graph.n_nodes)[graph.node2graph])
        node_index = (node_mask | single_node_mask).nonzero().squeeze(-1)
        compact_mapping[node_index] = ops.arange(len(node_index))
        node_in = compact_mapping[node_in]
        node_out = compact_mapping[node_out]
        graph = graph.subgraph(node_index)

        node_in_target = node_in - graph.cum_nodes + graph.n_nodes
        assert (node_in_target[valid] < graph.n_nodes[valid]).all() and (node_in_target[valid] >= 0).all()
        # node2 might be a new node
        node_out_target = ops.where(node_out == -1, self.atom2id[atom_out] + graph.n_nodes,
                                    node_out - graph.cum_nodes + graph.n_nodes)
        stop_target = ops.zeros(len(node_in_target))

        graph = graph[valid]
        node_in_target = node_in_target[valid]
        node_out_target = node_out_target[valid]
        bond_target = bond_target[valid]
        stop_target = stop_target[valid]

        assert (graph.n_edges % 2 == 0).all()
        # node / edge features may change because we mask some nodes / edges
        graph, feature_valid = self._update_molecule_feature(graph)

        return graph[feature_valid], node_in_target[feature_valid], node_out_target[feature_valid], \
            bond_target[feature_valid], stop_target[feature_valid]

    def all_stop(self, reactant, synthon):
        """_summary_

        Args:
            reactant (_type_): _description_
            synthon (_type_): _description_

        Returns:
            _type_: _description_
        """
        graph = reactant.clone()
        node_r2s, edge_r2s, is_new_node, is_new_edge, is_modified_edge, is_reaction_center = \
            self._get_reaction_feature(reactant, synthon)
        with graph.node():
            graph.node_r2s = node_r2s
            graph.is_new_node = is_new_node
            graph.is_reaction_center = is_reaction_center
        with graph.edge():
            graph.edge_r2s = edge_r2s
            graph.is_new_edge = is_new_edge
            graph.is_modified_edge = is_modified_edge

        node_in_target = ops.zeros(len(graph), dtype=ms.int64)
        node_out_target = ops.zeros_like(node_in_target)
        bond_target = ops.zeros_like(node_in_target)
        stop_target = ops.ones(len(graph))

        # keep consistent with other training data
        graph, feature_valid = self._update_molecule_feature(graph)

        return graph[feature_valid], node_in_target[feature_valid], node_out_target[feature_valid], \
            bond_target[feature_valid], stop_target[feature_valid]

    def construct(self, *batch):
        """_summary_

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        all_loss = 0
        metric = {}
        batch, _ = core.args_from_dict(batch)
        pred, target = self.predict_and_target(batch, all_loss, metric)
        node_in_pred, node_out_pred, bond_pred, stop_pred = pred
        node_in_target, node_out_target, bond_target, stop_target, size = target

        loss = functional.variadic_cross_entropy(node_in_pred, node_in_target, size, reduction="none")
        loss = functional.masked_mean(loss, stop_target == 0)
        metric["node in ce loss"] = loss
        all_loss += loss

        loss = functional.variadic_cross_entropy(node_out_pred, node_out_target, size, reduction="none")
        loss = functional.masked_mean(loss, stop_target == 0)
        metric["node out ce loss"] = loss
        all_loss += loss

        loss = ops.cross_entropy(bond_pred, bond_target, reduction="none")
        loss = functional.masked_mean(loss, stop_target == 0)
        metric["bond ce loss"] = loss
        all_loss += loss

        # Do we need to balance stop pred?
        loss = nn.BCELoss()(stop_pred, stop_target)
        metric["stop bce loss"] = loss
        all_loss += loss

        metric["total loss"] = all_loss
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
        node_in_pred, node_out_pred, bond_pred, stop_pred = pred
        node_in_target, node_out_target, bond_target, stop_target, size = target

        metric = {}

        node_in_acc = metrics.variadic_accuracy(node_in_pred, node_in_target, size)
        accuracy = functional.masked_mean(node_in_acc, stop_target == 0)
        metric["node in accuracy"] = accuracy

        node_out_acc = metrics.variadic_accuracy(node_out_pred, node_out_target, size)
        accuracy = functional.masked_mean(node_out_acc, stop_target == 0)
        metric["node out accuracy"] = accuracy

        bond_acc = (bond_pred.argmax(-1) == bond_target).astype(ms.float32)
        accuracy = functional.masked_mean(bond_acc, stop_target == 0)
        metric["bond accuracy"] = accuracy

        stop_acc = ((stop_pred > 0.5) == (stop_target > 0.5)).astype(ms.float32)
        metric["stop accuracy"] = stop_acc.mean()

        total_acc = (node_in_acc > 0.5) & (node_out_acc > 0.5) & (bond_acc > 0.5) & (stop_acc > 0.5)
        total_acc = ops.where(stop_target == 0, total_acc, stop_acc > 0.5).astype(ms.float32)
        metric["total accuracy"] = total_acc.mean()

        return metric

    def target(self, batch):
        """_summary_

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        reactant, synthon = batch["graph"]

        graph1, node_in_target1, node_out_target1, bond_target1, stop_target1 = self.all_edge(reactant, synthon)
        graph2, node_in_target2, node_out_target2, bond_target2, stop_target2 = self.all_stop(reactant, synthon)

        node_in_target = ops.concat([node_in_target1, node_in_target2])
        node_out_target = ops.concat([node_out_target1, node_out_target2])
        bond_target = ops.concat([bond_target1, bond_target2])
        stop_target = ops.concat([stop_target1, stop_target2])
        size = ops.concat([graph1.n_nodes, graph2.n_nodes])
        # add new atom candidates into the size of each graph
        size_ext = size + self.n_atom_type

        return node_in_target, node_out_target, bond_target, stop_target, size_ext

    def predict_reactant(self, batch, n_beam=10, max_prediction=20, max_step=20):
        """_summary_

        Args:
            batch (_type_): _description_
            n_beam (int, optional): _description_. Defaults to 10.
            max_prediction (int, optional): _description_. Defaults to 20.
            max_step (int, optional): _description_. Defaults to 20.

        Returns:
            _type_: _description_
        """
        if "synthon" in batch:
            synthon = batch["synthon"]
            synthon2product = ops.repeat(batch["n_synthon"])
            assert (synthon2product < len(batch["reaction"])).all()
            reaction = batch["reaction"][synthon2product]
        else:
            _, synthon = batch["graph"]
            reaction = batch["reaction"]

        # In any case, ensure that the synthon is a molecule rather than an ion
        # This is consistent across train/test routines in synthon completion
        synthon, feature_valid = self._update_molecule_feature(synthon)
        synthon = synthon[feature_valid]
        reaction = reaction[feature_valid]

        graph = synthon
        with graph.graph():
            # for convenience, because we need to manipulate graph a lot
            graph.reaction = reaction
            graph.synthon_id = ops.arange(len(graph))
            if not hasattr(graph, "logp"):
                graph.logp = ops.zeros(len(graph))
        with graph.node():
            graph.is_new_node = ops.zeros(graph.n_node, dtype=ops.bool_)
            graph.is_reaction_center = (graph.atom_map > 0) & \
                                       (graph.atom_map.expand_dims(-1) ==
                                        graph.reaction_center[graph.node2graph]).any(axis=-1)

        result = []
        n_prediction = ops.zeros(len(synthon), dtype=ms.int64)
        for i in range(max_step):
            logger.warning("action step: %d", i)
            logger.warning("batched beam size: %d", len(graph))
            # each candidate has #beam actions
            action, logp = self._topk_action(graph, n_beam)

            # each candidate is expanded to at most #beam (depending on validity) new candidates
            new_graph = self._apply_action(graph, action, logp)
            offset = -2 * (new_graph.logp.max() - new_graph.logp.min())
            key = new_graph.synthon_id * offset + new_graph.logp
            order = key.argsort(descending=True)
            new_graph = new_graph[order]

            n_candidate = new_graph.synthon_id.bincount(minlength=len(synthon))
            topk = functional.variadic_topk(new_graph.logp, n_candidate, n_beam)[1]
            topk_index = topk + (n_candidate.cumsum(0) - n_candidate).expand_dims(-1)
            topk_index = ops.unique(topk_index)
            new_graph = new_graph[topk_index]
            result.append(new_graph[new_graph.is_stopped])
            n_added = util.scatter_add(new_graph.is_stopped.astype(
                ms.int64), new_graph.synthon_id, n_axis=len(synthon))
            n_prediction += n_added

            # remove samples that already hit max prediction
            is_continue = (~new_graph.is_stopped) & (n_prediction[new_graph.synthon_id] < max_prediction)
            graph = new_graph[is_continue]
            if graph:
                break

        result = self._cat(result)
        # sort by synthon id
        order = result.synthon_id.argsort()
        result = result[order]

        # remove duplicate predictions
        is_duplicate = []
        synthon_id = -1
        for graph in result:
            if graph.synthon_id != synthon_id:
                synthon_id = graph.synthon_id
                smiles_set = set()
            smiles = graph.to_smiles(isomeric=False, atom_map=False, canonical=True)
            is_duplicate.append(smiles in smiles_set)
            smiles_set.add(smiles)
        is_duplicate = ms.Tensor(is_duplicate)
        result = result[~is_duplicate]
        n_prediction = result.synthon_id.bincount(minlength=len(synthon))

        # remove extra predictions
        topk = functional.variadic_topk(result.logp, n_prediction, max_prediction)[1]
        topk_index = topk + (n_prediction.cumsum(0) - n_prediction).expand_dims(-1)
        topk_index = topk_index.flatten(0)
        topk_index_shifted = ops.concat([-ops.ones(1, dtype=ms.int64), topk_index[:-1]])
        is_duplicate = topk_index == topk_index_shifted
        result = result[topk_index[~is_duplicate]]

        return result  # (< n_graph * max_prediction)

    def extend(self, data, n_xs, inputs, input2graph=None):
        """_summary_

        Args:
            data (_type_): _description_
            n_xs (_type_): _description_
            inputs (_type_): _description_
            input2graph (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if input2graph is None:
            n_input_per_graph = len(inputs) // len(n_xs)
            input2graph = ops.arange(len(n_xs)).expand_dims(-1)
            input2graph = input2graph.repeat(1, n_input_per_graph).flatten()
        n_inputs = input2graph.bincount(minlength=len(n_xs))
        new_n_xs = n_xs + n_inputs
        new_cum_xs = new_n_xs.cumsum(0)
        new_n_x = new_cum_xs[-1]
        new_data = ops.zeros(new_n_x, *data.shape[1:], dtype=data.dtype)
        starts = new_cum_xs - new_n_xs
        ends = starts + n_xs
        index = functional.multi_slice_mask(starts, ends, new_n_x)
        new_data[index] = data
        new_data[~index] = inputs
        return new_data, new_n_xs

    def predict_and_target(self, batch, all_loss=None, metric=None):
        """_summary_

        Args:
            batch (_type_): _description_
            all_loss (_type_, optional): _description_. Defaults to None.
            metric (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        reactant, synthon = batch["graph"]
        reactant = reactant.clone()
        with reactant.graph():
            reactant.reaction = batch["reaction"]

        graph1, node_in_target1, node_out_target1, bond_target1, stop_target1 = self.all_edge(reactant, synthon)
        graph2, node_in_target2, node_out_target2, bond_target2, stop_target2 = self.all_stop(reactant, synthon)

        graph = self._cat([graph1, graph2])

        node_in_target = ops.concat([node_in_target1, node_in_target2])
        node_out_target = ops.concat([node_out_target1, node_out_target2])
        bond_target = ops.concat([bond_target1, bond_target2])
        stop_target = ops.concat([stop_target1, stop_target2])
        size = graph.n_nodes
        # add new atom candidates into the size of each graph
        size_ext = size + self.n_atom_type

        synthon_feature = ops.stack([graph.is_new_node, graph.is_reaction_center], axis=-1).astype(ms.float32)
        node_feature = graph.node_feature.astype(ms.float32) + self.input_linear(synthon_feature)
        output = self.model(graph, node_feature, all_loss, metric)

        node_feature = [output["node_feature"]]
        graph_feature = []
        for feature_ in sorted(self.feature):
            if feature_ == "reaction":
                reaction_feature = ops.zeros((len(graph), self.n_reaction), dtype=ops.float32)
                reaction_feature.scatter_(1, graph.reaction.expand_dims(-1), 1)
                graph_feature.append(reaction_feature)
            elif feature_ == "graph":
                graph_feature.append(output["graph_feature"])
            elif feature_ == "atom":
                node_feature.append(graph.node_feature)
            else:
                raise ValueError(f"Unknown feature `{feature_}`")

        graph_feature = ops.concat(graph_feature, axis=-1)
        # inherit graph features
        node_feature.append(graph_feature[graph.node2graph])
        node_feature = ops.concat(node_feature, axis=-1)

        new_node_feature = self.new_atom_feature.weight.repeat(len(graph), 1)
        new_graph_feature = graph_feature.expand_dims(1).repeat(1, self.n_atom_type, 1).flatten(0, 1)
        new_node_feature = ops.concat([new_node_feature, new_graph_feature], axis=-1)
        node_feature, n_nodes_ext = self.extend(node_feature, graph.n_nodes, new_node_feature)
        assert (n_nodes_ext == size_ext).all()

        node2graph_ext = ops.repeat(n_nodes_ext)
        cum_nodes_ext = n_nodes_ext.cumsum(0)
        starts = cum_nodes_ext - n_nodes_ext + graph.n_nodes
        ends = cum_nodes_ext
        is_new_node = functional.multi_slice_mask(starts, ends, cum_nodes_ext[-1])

        node_in = node_in_target + cum_nodes_ext - n_nodes_ext
        node_out = node_out_target + cum_nodes_ext - n_nodes_ext
        edge = ops.stack([node_in, node_out], axis=-1)

        node_out_feature = ops.concat([node_feature[node_in][node2graph_ext], node_feature], axis=-1)
        bond_feature = node_feature[edge].flatten(-2)
        node_in_pred = self.node_in_mlp(node_feature).squeeze(-1)
        node_out_pred = self.node_out_mlp(node_out_feature).squeeze(-1)
        bond_pred = self.bond_mlp(bond_feature).squeeze(-1)
        stop_pred = self.stop_mlp(graph_feature).squeeze(-1)

        infinity = ms.Tensor(float("inf"))
        # mask out node-in prediction on new atoms
        node_in_pred[is_new_node] = -infinity
        # mask out node-out prediction on self-loops
        node_out_pred[node_in] = -infinity

        return (node_in_pred, node_out_pred, bond_pred, stop_pred), \
               (node_in_target, node_out_target, bond_target, stop_target, size_ext)

    def _update_molecule_feature(self, graphs):
        """_summary_

        Args:
            graphs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # This function is very slow
        graphs = graphs.ion_to_molecule()
        mols = graphs.to_molecule(ignore_error=True)
        valid = [mol is not None for mol in mols]
        valid = ms.Tensor(valid)
        new_graphs = type(graphs).from_molecule(mols, **self.feature_kwargs)

        node_feature = ops.zeros(graphs.n_node, *new_graphs.node_feature.shape[1:],
                                 dtype=new_graphs.node_feature.dtype)
        edge_feature = ops.zeros(graphs.n_edge, *new_graphs.edge_feature.shape[1:],
                                 dtype=new_graphs.edge_feature.dtype)
        bond_type = ops.zeros_like(graphs.bond_type)
        node_mask = valid[graphs.node2graph]
        edge_mask = valid[graphs.edge2graph]
        node_feature[node_mask] = new_graphs.node_feature.to(device=graphs.device)
        edge_feature[edge_mask] = new_graphs.edge_feature.to(device=graphs.device)
        bond_type[edge_mask] = new_graphs.bond_type.to(device=graphs.device)

        with graphs.node():
            graphs.node_feature = node_feature
        with graphs.edge():
            graphs.edge_feature = edge_feature
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
            n_max_x = n_xs.max()
            lengths = ops.arange(0, n_max_x, 2)

        pack_offsets = ops.arange(len(lengths)) * cum_xs[-1]
        # starts, lengths, ends: (n_repeat, n_graph)
        starts = starts.expand_dims(0) + pack_offsets.expand_dims(-1)
        valid = lengths.expand_dims(-1) <= n_xs.expand_dims(0) - 2
        lengths = ops.min(lengths.expand_dims(-1), n_xs.expand_dims(0) - 2).clamp(0)
        ends = starts + lengths

        starts = starts.flatten()
        ends = ends.flatten()
        valid = valid.flatten()

        return starts, ends, valid

    def _get_reaction_feature(self, reactant, synthon):
        """_summary_

        Args:
            reactant (_type_): _description_
            synthon (_type_): _description_
        """
        def get_edge_map(graph, n_nodes):
            """_summary_

            Args:
                graph (_type_): _description_
                n_nodes (_type_): _description_

            Returns:
                _type_: _description_
            """
            node_in, node_out = graph.edge_list.t()[:2]
            node_in2id = graph.atom_map[node_in]
            node_out2id = graph.atom_map[node_out]
            edge_map = node_in2id * n_nodes[graph.edge2graph] + node_out2id
            # edges containing any unmapped node is considered to be unmapped
            edge_map[(node_in2id == 0) | (node_out2id == 0)] = 0
            return edge_map

        def get_mapping(reactant_x, synthon_x, reactant_x2graph, synthon_x2graph):
            """_summary_

            Args:
                reactant_x (_type_): _description_
                synthon_x (_type_): _description_
                reactant_x2graph (_type_): _description_
                synthon_x2graph (_type_): _description_

            Returns:
                _type_: _description_
            """
            n_xs = util.scatter_max(reactant_x, reactant_x2graph)[0]
            n_xs = n_xs.clamp(0) + 1
            cum_xs = n_xs.cumsum(0)
            offset = cum_xs - n_xs
            reactant2id = reactant_x + offset[reactant_x2graph]
            synthon2id = synthon_x + offset[synthon_x2graph]
            assert synthon2id.min() > 0
            id2synthon = -ops.ones(cum_xs[-1], dtype=ms.int64)
            id2synthon[synthon2id] = ops.arange(len(synthon2id))
            reactant2synthon = id2synthon[reactant2id]

            return reactant2synthon

        # reactant & synthon may have different number of nodes
        assert (reactant.n_nodes >= synthon.n_nodes).all()
        reactant_edge_map = get_edge_map(reactant, reactant.n_nodes)
        synthon_edge_map = get_edge_map(synthon, reactant.n_nodes)

        node_r2s = get_mapping(reactant.atom_map, synthon.atom_map, reactant.node2graph, synthon.node2graph)
        edge_r2s = get_mapping(reactant_edge_map, synthon_edge_map, reactant.edge2graph, synthon.edge2graph)

        is_new_node = node_r2s == -1
        is_new_edge = edge_r2s == -1
        is_modified_edge = (edge_r2s != -1) & (reactant.bond_type != synthon.bond_type[edge_r2s])
        is_reaction_center = (reactant.atom_map > 0) & \
                             (reactant.atom_map.expand_dims(-1) ==
                              reactant.reaction_center[reactant.node2graph]).any(axis=-1)

        return node_r2s, edge_r2s, is_new_node, is_new_edge, is_modified_edge, is_reaction_center

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

    def _topk_action(self, graph, k):
        """_summary_

        Args:
            graph (_type_): _description_
            k (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        synthon_feature = ops.stack([graph.is_new_node, graph.is_reaction_center], axis=-1).astype(ms.float32)
        node_feature = graph.node_feature.astype(ms.float32) + self.input_linear(synthon_feature)
        output = self.model(graph, node_feature)

        node_feature = [output["node_feature"]]
        graph_feature = []
        for feature_ in sorted(self.feature):
            if feature_ == "reaction":
                reaction_feature = ops.zeros((len(graph), self.n_reaction), dtype=ops.float32)
                reaction_feature.scatter_(1, graph.reaction.expand_dims(-1), 1)
                graph_feature.append(reaction_feature)
            elif feature_ == "graph":
                graph_feature.append(output["graph_feature"])
            elif feature_ == "atom":
                node_feature.append(graph.node_feature.astype(ms.float32))
            else:
                raise ValueError(f"Unknown feature `{feature_}`")

        graph_feature = ops.concat(graph_feature, axis=-1)
        # inherit graph features
        node_feature.append(graph_feature[graph.node2graph])
        node_feature = ops.concat(node_feature, axis=-1)

        new_node_feature = self.new_atom_feature.weight.repeat(len(graph), 1)
        new_graph_feature = graph_feature.expand_dims(1).repeat(1, self.n_atom_type, 1).flatten(0, 1)
        new_node_feature = ops.concat([new_node_feature, new_graph_feature], axis=-1)
        node_feature, n_nodes_ext = self.extend(node_feature, graph.n_nodes, new_node_feature)

        node2graph_ext = ops.repeat(n_nodes_ext)
        cum_nodes_ext = n_nodes_ext.cumsum(0)
        starts = cum_nodes_ext - n_nodes_ext + graph.n_nodes
        ends = cum_nodes_ext
        is_new_node = functional.multi_slice_mask(starts, ends, cum_nodes_ext[-1])
        infinity = float("inf")

        node_in_pred = self.node_in_mlp(node_feature).squeeze(-1)
        stop_pred = self.stop_mlp(graph_feature).squeeze(-1)

        # mask out node-in prediction on new atoms
        node_in_pred[is_new_node] = -infinity
        node_in_logp = functional.variadic_log_softmax(node_in_pred, n_nodes_ext)  # (n_node,)
        stop_logp = ops.logsigmoid(stop_pred)
        act_logp = ops.logsigmoid(-stop_pred)
        node_in_topk = functional.variadic_topk(node_in_logp, n_nodes_ext, k)[1]
        assert (node_in_topk >= 0).all() and (node_in_topk < n_nodes_ext.expand_dims(-1)).all()
        node_in = node_in_topk + (cum_nodes_ext - n_nodes_ext).expand_dims(-1)  # (n_graph, k)

        # The shape of node_out_feature is (n_node, node_in_k, feature_dim)
        node_out_feature = ops.concat([node_feature[node_in][node2graph_ext],
                                       node_feature.expand_dims(1).expand(-1, k, -1)], axis=-1)
        node_out_pred = self.node_out_mlp(node_out_feature).squeeze(-1)
        # mask out node-out prediction on self-loops
        node_out_pred.scatter_(0, node_in, -infinity)
        # The shape is (n_node, node_in_k)
        node_out_logp = functional.variadic_log_softmax(node_out_pred, n_nodes_ext)
        # The shape of node_out_topk is (n_graph, node_out_k, node_in_k)
        node_out_topk = functional.variadic_topk(node_out_logp, n_nodes_ext, k)[1]
        assert (node_out_topk >= 0).all() and (node_out_topk < n_nodes_ext.view(-1, 1, 1)).all()
        node_out = node_out_topk + (cum_nodes_ext - n_nodes_ext).view(-1, 1, 1)

        # The shape of bond_feature is (n_graph, node_out_k, node_in_k, feature_dim * 2)
        edge = ops.stack([node_in.expand_dims(1).expand_as(node_out), node_out], axis=-1)
        bond_feature = node_feature[edge].flatten(-2)
        bond_pred = self.bond_mlp(bond_feature).squeeze(-1)
        bond_logp = ops.log_softmax(bond_pred, axis=-1)  # (n_graph, node_out_k, node_in_k, n_relation)
        bond_type = ops.arange(bond_pred.shape[-1])
        bond_type = bond_type.view(1, 1, 1, -1).expand_as(bond_logp)

        # The shape of node_in_logp is (n_graph, node_out_k, node_in_k, n_relation)
        node_in_logp = node_in_logp.gather(0, node_in.flatten(0, 1)).view(-1, 1, k, 1)
        node_out_logp = node_out_logp.gather(0, node_out.flatten(0, 1)).view(-1, k, k, 1)
        act_logp = act_logp.view(-1, 1, 1, 1)
        logp = node_in_logp + node_out_logp + bond_logp + act_logp

        # The shape is (n_graph, node_out_k, node_in_k, n_relation, 4)
        node_in_topk = node_in_topk.view(-1, 1, k, 1).expand_as(logp)
        node_out_topk = node_out_topk.view(-1, k, k, 1).expand_as(logp)
        action = ops.stack([node_in_topk, node_out_topk, bond_type, ops.zeros_like(bond_type)], axis=-1)

        # add stop action
        logp = ops.concat([logp.flatten(1), stop_logp.expand_dims(-1)], axis=1)
        stop = ms.Tensor([0, 0, 0, 1])
        stop = stop.view(1, 1, -1).expand(len(graph), -1, -1)
        action = ops.concat([action.flatten(1, -2), stop], axis=1)
        topk = logp.topk(k, axis=-1)[1]

        return action.gather(1, topk.expand_dims(-1).expand(-1, -1, 4)), logp.gather(1, topk)

    def _apply_action(self, graph, action, logp):
        """_summary_

        Args:
            graph (_type_): _description_
            action (_type_): _description_
            logp (_type_): _description_

        Returns:
            _type_: _description_
        """
        # only support non-variadic k-actions
        assert len(graph) == len(action)
        n_action = action.shape[1]

        graph = graph.repeat(n_action)

        action = action.flatten(0, 1)  # (n_graph * k, 4)
        logp = logp.flatten(0, 1)  # (n_graph * k)
        new_node_in, new_node_out, new_bond_type, stop = action.t()

        # add new nodes
        has_new_node = (new_node_out >= graph.n_nodes) & (stop == 0)
        new_atom_id = (new_node_out - graph.n_nodes)[has_new_node]
        new_atom_type = self.id2atom[new_atom_id]
        is_new_node = ops.ones(len(new_atom_type), dtype=ms.bool_)
        is_reaction_center = ops.zeros(len(new_atom_type), dtype=ms.bool_)
        atom_type, n_nodes = functional.extend(graph.atom_type, graph.n_nodes, new_atom_type, has_new_node)
        is_new_node = functional.extend(graph.is_new_node, graph.n_nodes, is_new_node, has_new_node)[0]
        is_reaction_center = functional.extend(
            graph.is_reaction_center, graph.n_nodes, is_reaction_center, has_new_node)[0]

        # cast to regular node ids
        new_node_out = ops.where(has_new_node, graph.n_nodes, new_node_out)

        # modify edges
        new_edge = ops.stack([new_node_in, new_node_out], axis=-1)
        edge_list = graph.edge_list.clone()
        bond_type = graph.bond_type.clone()
        edge_list[:, :2] -= graph.offsets.expand_dims(-1)
        is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(axis=-1) & \
                           (stop[graph.edge2graph] == 0)
        has_modified_edge = util.scatter_max(is_modified_edge.astype(
            ms.int64), graph.edge2graph, n_axis=len(graph))[0] > 0
        bond_type[is_modified_edge] = new_bond_type[has_modified_edge]
        edge_list[is_modified_edge, 2] = new_bond_type[has_modified_edge]
        # modify reverse edges
        new_edge = new_edge.flip(-1)
        is_modified_edge = (edge_list[:, :2] == new_edge[graph.edge2graph]).all(axis=-1) & \
                           (stop[graph.edge2graph] == 0)
        bond_type[is_modified_edge] = new_bond_type[has_modified_edge]
        edge_list[is_modified_edge, 2] = new_bond_type[has_modified_edge]

        # add new edges
        has_new_edge = (~has_modified_edge) & (stop == 0)
        new_edge_list = ops.stack([new_node_in, new_node_out, new_bond_type], axis=-1)[has_new_edge]
        bond_type = functional.extend(bond_type, graph.n_edges, new_bond_type[has_new_edge], has_new_edge)[0]
        edge_list, n_edges = functional.extend(edge_list, graph.n_edges, new_edge_list, has_new_edge)
        # add reverse edges
        new_edge_list = ops.stack([new_node_out, new_node_in, new_bond_type], axis=-1)[has_new_edge]
        bond_type = functional.extend(bond_type, n_edges, new_bond_type[has_new_edge], has_new_edge)[0]
        edge_list, n_edges = functional.extend(edge_list, n_edges, new_edge_list, has_new_edge)

        logp = logp + graph.logp

        # inherit attributes
        data_dict = graph.data_dict
        meta_dict = graph.meta_dict
        for key in ["atom_type", "bond_type", "is_new_node", "is_reaction_center", "logp"]:
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
                                is_new_node=is_new_node, is_reaction_center=is_reaction_center, logp=logp,
                                meta_dict=meta_dict, **data_dict)
        with new_graph.graph():
            new_graph.is_stopped = stop == 1
        valid = logp > float("-inf")
        new_graph = new_graph[valid]

        new_graph, feature_valid = self._update_molecule_feature(new_graph)
        return new_graph[feature_valid]
