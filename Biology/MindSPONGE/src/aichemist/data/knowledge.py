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
from .graph import Graph
from .. import core
from .. import utils

logger = logging.getLogger(__name__)


class KnowledgeGraphSet(core.BaseDataset):
    """
    Knowledge of graph dataset.

    The whole dataset contains one knowledge graph.

    Args:
        verbose (int, optional):            if equals to 1, the detailed information will be output during the
                                            data process. Defaults to 0.
        graph (Graph, optional):            Graph object that include all of knowledge data. Defaults to None.
        n_samples (array_like, optional):   Number of samples for each subset. Defaults to None.
        entity_voc (dict, optional):        The vocabulary of entities. The key is index and the value is entity name.
                                            Defaults to None.
        relation_voc (dict, optional):      The vocabulary of relations. he key is index and the value is relation name.
                                            Defaults to None.
        inv_entity_voc (dict, optional):    The inverse vocabulary of entities. The key is entity name and the
                                            value is index. Defaults to None.
        inv_relation_voc (dict, optional):  The inverse vocabulary of relations. The key is relation name and the
                                            value is index. Defaults to None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    _caches = ['edges', 'edge_type']

    def __init__(self,
                 verbose=0,
                 graph=None,
                 n_samples=None,
                 entity_voc=None,
                 relation_voc=None,
                 inv_entity_voc=None,
                 inv_relation_voc=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.edges = graph.edges if graph is not None else []
        self.edge_type = graph.edge_type if graph is not None else []
        self.verbose = verbose
        self.n_samples = n_samples
        self.entity_voc = entity_voc
        self.relation_voc = relation_voc
        self.inv_entity_voc = inv_entity_voc
        self.inv_relation_voc = inv_relation_voc

    def __len__(self):
        return self.graph.n_edge

    def __repr__(self):
        lines = f"#node: {self.graph.n_node}\n" + \
                f"#relation: {self.n_relation}\n" + \
                f"#edge: {self.graph.n_edge}"
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

    def subset(self, indices):
        subset = super().subset(indices)
        subset.graph = subset.graph.mask_edge(indices)
        return subset

    def load_file(self, fnames):
        """
        Load data from files.

        Args:
            fnames (List[str]): the path of files
        """
        if isinstance(fnames, str):
            fnames = [fnames]
        df = []
        for fname in fnames:
            if fname.split('.')[-1] in self._seps:
                sep = self._seps.get(fname.split('.')[-1])
                df.append(pd.read_table(fname, sep=sep, names=['source', 'relation', 'target']))
        self.n_samples = [len(d) for d in df]
        df = pd.concat(df)
        relation_voc = np.unique(df['relation'].values)
        entity_voc = np.unique(df[['source', 'target']].values.reshape(-1))
        self.entity_voc, self.inv_entity_voc = _standarize_voc(entity_voc, None)
        self.relation_voc, self.inv_relation_voc = _standarize_voc(relation_voc, None)
        sources = [self.inv_entity_voc[i] for i in df['source']]
        targets = [self.inv_entity_voc[i] for i in df['target']]
        self.edges = np.array([sources, targets]).T
        self.edge_type = np.array([self.inv_relation_voc[i] for i in df['relation']])
        n_node = len(self.entity_voc)
        n_relation = len(self.relation_voc)
        self.graph = data.Graph(edges=self.edges.T, n_node=n_node, n_relation=n_relation, edge_type=self.edge_type)
        return self


class KnowledgeNodeSet(core.BaseDataset):
    """
    Node classification dataset.

    The whole dataset contains one graph, where each node has its own node feature and label.

    Args:
        verbose (int, optional):        if equals to 1, the detailed information will be output during the data process.
                                        Defaults to 0.
        graph (Graph, optional):        Graph object that include all of knowledge data. Defaults to None.
        node_voc (dict, optional):      The vocabulary of entities. The key is index and the value is node name.
                                        Defaults to None.
        inv_node_voc (dict, optional):  The inverse vocabulary of nodes. he key is index and the value is node name.
                                        Defaults to None.
        label_voc (dict, optional):     The vocabulary of labels. The key is index and the value is label name.
                                        Defaults to None.
        inv_label_voc (dict, optional): The inverse vocabulary of labels. The key is label name and the
                                        value is index. Defaults to None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    _caches = ['node_feat', 'node_type']

    def __init__(self,
                 verbose=0,
                 graph=None,
                 node_voc=None,
                 inv_node_voc=None,
                 label_voc=None,
                 inv_label_voc=None,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.verbose = verbose
        self.node_voc = node_voc
        self.inv_node_voc = inv_node_voc
        self.label_voc = label_voc
        self.inv_label_voc = inv_label_voc
        self.graph = graph
        self.node_feat = [] if self.graph is None else graph.node_feat
        self.node_type = [] if self.graph is None else graph.node_type

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

    def subset(self, indices):
        subset = super().subset(indices)
        subset.graph = subset.graph.mask_edge(indices)
        return subset

    def load_tsv(self, node_file, edge_file):
        """
        Load the edge list from a tsv file.

        Args:
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
            feature = [utils.literal_eval(f) for f in feature_tokens]
            label = inv_label_voc.get(label_token)
            node_feat.append(feature)
            node_type.append(label)

        edges = []

        df = pd.read_csv(edge_file)
        for _, tokens in df.iterrows():
            h_token, t_token = tokens
            if h_token not in inv_node_voc:
                inv_node_voc[h_token] = len(inv_node_voc)
            h = inv_node_voc.get(h_token)
            if t_token not in inv_node_voc:
                inv_node_voc[t_token] = len(inv_node_voc)
            t = inv_node_voc.get(t_token)
            edges.append((h, t))

        self.load_edge(edges, node_feat, node_type, inv_node_voc=inv_node_voc,
                       inv_label_voc=inv_label_voc)

    def load_edge(self, edges, node_feat, node_type, node_voc=None, inv_node_voc=None, label_voc=None,
                  inv_label_voc=None):
        """Load the graph data

        Args:
            edges (array_like):             The 2 x M matrix of edges. The first row is index of start nodes
                                            and the second row is the index of end nodes.
            node_feat (array_like):         The matrix of node features. The shape is (N, Fn).
            edge_type (array_like):         The matrix of edge types. The shape is (M, ).
            node_voc (dict, optional):      The vocabulary of entities. The key is index and the value is node name.
                                            Defaults to None.
            inv_node_voc (dict, optional):  The inverse vocabulary of nodes. he key is index and the value is node name.
                                            Defaults to None.
            label_voc (dict, optional):     The vocabulary of labels. The key is index and the value is label name.
                                            Defaults to None.
            inv_label_voc (dict, optional): The inverse vocabulary of labels. The key is label name and the
                                            value is index. Defaults to None.
        """
        self.node_voc, self.inv_node_voc = _standarize_voc(node_voc, inv_node_voc)
        self.label_voc, self.inv_label_voc = _standarize_voc(label_voc, inv_label_voc)

        if len(node_voc) > len(node_feat):
            logger.warning("Missing features & labels for %d / %d nodes", len(node_voc) - len(node_feat), len(node_voc))
            dummy_label = 0
            dummy_feature = [0] * len(node_feat[0])
            node_type += [dummy_label] * (len(node_voc) - len(node_feat))
            node_feat += [dummy_feature] * (len(node_voc) - len(node_feat))
        self.node_feat = np.array(node_feat)
        self.node_type = np.array(node_type)
        self.graph = Graph(edges=edges, n_node=len(node_voc),
                           node_feat=self.node_feat, node_type=self.node_type)


def _standarize_voc(voc, inv_voc):
    """
    Standarize the vocabulary.

    Args:
        voc (dict): vocabuary
        inv_voc (dict): inverse vocabulary that key and value is inversed from voc.

    Returns:
        voc (dict): vocabuary
        inv_voc (dict): inverse vocabuary
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
