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
knowledge
"""
import logging
import numpy as np
import pandas as pd

from .. import data
from . import graph
from .. import core
from .. import util

logger = logging.getLogger(__name__)


class KnowledgeGraphSet(core.DataLoader):
    """
    Knowledge of graph dataset.

    The whole dataset contains one knowledge graph.
    """
    _caches = []

    def __init__(self,
                 batch_size: int = 128,
                 shuffle=False,
                 verbose=0,
                 **kwargs) -> None:
        super().__init__(batch_size, verbose, shuffle, **kwargs)

    def __getitem__(self, index):
        index = self._standarize_index(index)
        index = self._order[index].tolist()
        sources, targets = self.graph.edges[:, index]
        return sources, targets, self.graph.edge_type[index]

    def __next__(self):
        batch = super().__next__()
        return [self.graph] + batch

    def __len__(self):
        return self.graph.n_edge

    def __repr__(self):
        lines = f"#node: {self.graph.n_node}\n" + \
                f"#relation: {self.n_relation}\n" + \
                f"#edge: {self.grpah.n_edge}"
        return f"{self.__class__.__name__}(\n  {lines}n)"

    @property
    def n_entity(self):
        """Number of entities."""
        return self.graph.n_node

    @property
    def n_triplet(self):
        """Number of triplets."""
        return self.graph.n_edge

    @property
    def n_relation(self):
        """Number of relations."""
        return self.graph.n_relation

    def initialize(self, **kwargs):
        """_summary_
        """
        super().initialize(**kwargs)
        if hasattr(self, 'graph'):
            self._order = np.arange(self.graph.n_edge)

    def load_file(self, fnames):
        """_summary_

        Args:
            fnames (_type_): _description_
        """
        if isinstance(fnames, str):
            fnames = [fnames]
        df = []
        for fname in fnames:
            if fname.split('.')[-1] in self._seps:
                sep = self._seps.get(fname.split('.')[-1])
                df.append(pd.read_table(fname, sep=sep, names=['source', 'relation', 'target']))
        n_samples = [len(d) for d in df]
        df = pd.concat(df)
        relation_voc = np.unique(df['relation'].values)
        entity_voc = np.unique(df[['source', 'target']].values.reshape(-1))
        entity_voc, inv_entity_voc = _standarize_voc(entity_voc, None)
        relation_voc, inv_relation_voc = _standarize_voc(relation_voc, None)
        sources = [inv_entity_voc[i] for i in df['source']]
        targets = [inv_entity_voc[i] for i in df['target']]
        edges = np.array([sources, targets])
        edge_type = np.array([inv_relation_voc[i] for i in df['relation']])
        n_node = len(entity_voc)
        n_relation = len(relation_voc)
        graph_ = data.Graph(edges, n_node=n_node, n_relation=n_relation, edge_type=edge_type)
        self.initialize(entity_voc=entity_voc,
                        relation_voc=relation_voc,
                        inv_entity_voc=inv_entity_voc,
                        inv_relation_voc=inv_relation_voc,
                        n_samples=n_samples, graph=graph_)

    def split(self, ratios=None):
        """_summary_

        Args:
            ratio (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        ratios = self.n_samples if ratios is None else self.n_samples
        return super().split(self.n_samples)

    def subset(self, indices):
        """_summary_

        Args:
            indices (_type_): _description_

        Returns:
            _type_: _description_
        """
        subset = super().subset(indices)
        subset.graph = subset.graph.mask_edge(indices)
        return subset


class KnowledgeNodeSet(core.DataLoader):
    """
    Node classification dataset.

    The whole dataset contains one graph, where each node has its own node feature and label.
    """

    def __getitem__(self, index):
        return self.graph.node_feat[index], self.graph.node_type[index]

    def __len__(self):
        return self.n_node

    def __repr__(self):
        lines = f"#node: {self.n_node}\n" + \
                f"#edge: {self.n_edge}\n" + \
                f"#class: {len(self.type_voc)}\n"
        return f"{self.__class__.__name__}(\n  {lines}\n)"

    @property
    def n_node(self):
        """Number of nodes."""
        return self.graph.n_node

    @property
    def n_edge(self):
        """Number of edges."""
        return self.graph.n_edge

    @property
    def node_feature_dim(self):
        """Dimension of node features."""
        return self.graph.node_feature.shape[-1]

    def load_tsv(self, node_file, edge_file):
        """
        Load the edge list from a tsv file.

        Parameters:
            node_file (str): node feature and label file
            edge_file (str): edge list file
            verbose (int, optional): output verbose level
        """
        inv_node_voc = {}
        inv_label_voc = {}
        node_feat = []
        node_type = []

        df = pd.read_csv(node_file)
        for _, tokens in df.iterrows():
            node_token = tokens[0]
            feature_tokens = tokens[1: -1]
            label_token = tokens[-1]
            inv_node_voc[node_token] = len(inv_node_voc)
            if label_token not in inv_label_voc:
                inv_label_voc[label_token] = len(inv_label_voc)
            feature = [util.literal_eval(f) for f in feature_tokens]
            label = inv_label_voc[label_token]
            node_feat.append(feature)
            node_type.append(label)

        edge_list = []

        df = pd.read_csv(edge_file)
        for _, tokens in df.iterrows():
            h_token, t_token = tokens
            if h_token not in inv_node_voc:
                inv_node_voc[h_token] = len(inv_node_voc)
            h = inv_node_voc[h_token]
            if t_token not in inv_node_voc:
                inv_node_voc[t_token] = len(inv_node_voc)
            t = inv_node_voc[t_token]
            edge_list.append((h, t))

        self.load_edge(edge_list, node_feat, node_type, inv_node_voc=inv_node_voc,
                       inv_label_voc=inv_label_voc)

    def load_edge(self, edges, node_feat, node_type, node_voc=None, inv_node_voc=None, label_voc=None,
                  inv_label_voc=None):
        """_summary_

        Args:
            edges (_type_): _description_
            node_feat (_type_): _description_
            node_type (_type_): _description_
            node_voc (_type_, optional): _description_. Defaults to None.
            inv_node_voc (_type_, optional): _description_. Defaults to None.
            label_voc (_type_, optional): _description_. Defaults to None.
            inv_label_voc (_type_, optional): _description_. Defaults to None.
        """
        node_voc, inv_node_voc = _standarize_voc(node_voc, inv_node_voc)
        label_voc, inv_label_voc = _standarize_voc(label_voc, inv_label_voc)

        if len(node_voc) > len(node_feat):
            logger.warning("Missing features & labels for %d / %d nodes", len(node_voc) - len(node_feat), len(node_voc))
            dummy_label = 0
            dummy_feature = [0] * len(node_feat[0])
            node_type += [dummy_label] * (len(node_voc) - len(node_feat))
            node_feat += [dummy_feature] * (len(node_voc) - len(node_feat))

        graph_data = graph.Graph(edges, n_node=len(node_voc), node_feat=node_feat, node_type=node_type)
        setattr(self, 'graph', graph_data)
        setattr(self, 'node_type', node_type)
        setattr(self, 'node_voc', node_voc)
        setattr(self, 'inv_node_voc', inv_node_voc)
        setattr(self, 'type_voc', label_voc)
        setattr(self, 'inv_node_voc', inv_label_voc)


def _standarize_voc(voc, inv_voc):
    """_summary_

    Args:
        voc (_type_): _description_
        inv_voc (_type_): _description_

    Returns:
        _type_: _description_
    """
    if voc is not None:
        if isinstance(voc, dict):
            assert set(voc.keys()) == set(range(len(voc))), "Vocabulary keys should be consecutive numbers"
            voc = [voc[k] for k in range(len(voc))]
        if inv_voc is None:
            inv_voc = {v: i for i, v in enumerate(voc)}
    if inv_voc is not None:
        assert set(inv_voc.values()) == set(range(len(inv_voc))), \
            "Inverse vocabulary values should be consecutive numbers"
        if voc is None:
            voc = sorted(inv_voc, key=lambda k: inv_voc[k])
    return voc, inv_voc
