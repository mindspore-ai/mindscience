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
graph
"""

import mindspore as ms
from mindspore import ops
import numpy as np
from scipy import spatial

from .. import util
from . import feature
from .. import core
from ..core import Registry as R


@R.register('data.Graph')
class Graph(core.MetaData):
    """_summary_

    Args:
        core (_type_): _description_

    Returns:
        _type_: _description_
    """
    _caches = {'node_': 'ndata', 'edge_': 'edata', 'graph_': 'gdata'}

    def __init__(self,
                 edges=None,
                 ndata=None,
                 edata=None,
                 gdata=None,
                 n_relation=None,
                 n_node=None,
                 **kwargs):
        # The shape of edges is N * [h, t]
        super().__init__(**kwargs)
        self.edges = edges
        self.ndata = self._check_cache(ndata)
        self.edata = self._check_cache(edata)
        self.gdata = self._check_cache(gdata)
        self.n_relation = self._maybe_n_node(self.edge_type) if n_relation is None else n_relation
        self.n_node = self._maybe_n_node(self.edges) if n_node is None else n_node
        self.n_edge = edges.shape[1]
        for key, value in kwargs.items():
            setattr(self, key, value)
        if 'weight' not in self.edata:
            self.edata['weight'] = np.ones(edges.shape[1])

    def __getitem__(self, index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        # why do we check tuple?
        # case 1: x[0, 1] is parsed as (0, 1)
        # case 2: x[[0, 1]] is parsed as [0, 1]
        if not isinstance(index_, tuple):
            index_ = (index,)
        index_ = list(index)

        while len(index_) < 2:
            index_.append(slice(None))
        assert len(index_) <= 2

        is_valid = np.all([isinstance(axis_index, int) for axis_index in index_])
        if is_valid:
            return self.get_edge(index_)
        edges = self.edges.copy()
        for i, axis_index in enumerate(index_):
            axis_index = self._standarize_index(axis_index, self.n_node)
            mapping = - np.ones(self.n_node, dtype=np.int32)
            mapping[axis_index] = axis_index
            edges[:, i] = mapping[edges[:, i]]
        edge_index = (edges >= 0).all(axis=-1)

        return self.mask_edge(edge_index)

    def __len__(self):
        return 1

    @property
    def shape(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.size()

    @property
    def batch_size(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return 1

    @classmethod
    def from_dense(cls, adjacency: np.ndarray, ndata=None, edata=None):
        """_summary_

        Args:
            adjacency (np.ndarray): _description_
            ndata (_type_, optional): _description_. Defaults to None.
            edata (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        assert adjacency.shape[0] == adjacency.shape[1]
        edges = adjacency.nonzero().T
        n_node = adjacency.shape[0]
        n_relation = adjacency.shape[2] if adjacency.ndim > 2 else None
        if edata is not None:
            edata = edata[tuple(edges)]
        edata = {'feat': edata}
        edata['weight'] = adjacency[adjacency != 0]
        return cls(edges, ndata=ndata, edata=edata, n_node=n_node, n_relation=n_relation)

    @classmethod
    def knn_graph(cls, coord, max_neighbors=None, cutoff=None, **kwargs):
        """_summary_

        Args:
            coord (_type_): _description_
            max_neighbors (_type_, optional): _description_. Defaults to None.
            cutoff (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        dist = spatial.distance.cdist(coord, coord)

        if max_neighbors is None:
            edges = np.stack(np.nonzero(dist < cutoff), axis=1)
        else:
            neighbors = np.argsort(dist, axis=1)
            sorted_dist = np.sort(dist, axis=1)
            if max_neighbors < len(coord):
                neighbors = neighbors[:, 1:max_neighbors]
            mask = sorted_dist < cutoff
            edge_in = np.nonzero(mask)[:, 0].astype(ms.int32)
            edge_out = neighbors[mask]
            edges = np.stack([edge_in, edge_out], axis=1)
        dist_list = dist[tuple(edges)]
        edata = feature.distance(dist_list=dist_list, **kwargs)
        ndata = {'coord': coord}
        graph = cls(edges=edges.T, n_node=dist.shape[0], edata=edata, ndata=ndata)
        return graph

    @classmethod
    def pack(cls, graphs: list):
        """_summary_

        Args:
            graphs (list): _description_

        Returns:
            _type_: _description_
        """
        n_relation = -1
        ndata = None if graphs[0].ndata is None else {key: [] for key in graphs[0].ndata}
        edata = None if graphs[0].edata is None else {key: [] for key in graphs[0].edata}
        gdata = None if graphs[0].gdata is None else {key: [] for key in graphs[0].gdata}

        for cache_ in ['ndata', 'edata', 'gdata']:
            cache = locals()[cache_]
            for key in cache:
                value = []
                for g in graphs:
                    param = getattr(g, cache_)
                    if key in param:
                        value.append(param[key])
                cache[key] = np.concatenate(value)
        assert np.array([g.n_relation == graphs[0].n_relation for g in graphs]).all()
        n_nodes = np.array([g.n_node for g in graphs])
        n_edges = np.array([g.n_edge for g in graphs])
        edges = np.concatenate([g.edges for g in graphs], axis=1)
        return cls.batch_type(edges, n_nodes=n_nodes, n_edges=n_edges, n_relation=n_relation,
                              ndata=ndata, edata=edata, gdata=gdata)

    def connected_components(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        node_in, node_out = self.edges
        order = np.arange(self.n_node)
        node_in = np.concatenate([node_in, node_out, order])
        node_out = np.concatenate([node_out, node_in, order])

        min_neighbor = np.arange(self.n_node)
        last = np.zeros_like(min_neighbor)
        while not np.equal(min_neighbor, last):
            last = min_neighbor
            min_neighbor = util.scatter_min(min_neighbor[node_out], node_in, n_axis=self.n_node)
        anchor = np.unique(min_neighbor)
        n_cc = self.node2graph()[anchor].bincount(minlength=self.batch_size)
        return self.split(min_neighbor), n_cc

    def split(self, node2graph: np.ndarray):
        """_summary_

        Args:
            node2graph (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        _, node2graph = np.unique(node2graph, return_inverse=True)
        n_graph = node2graph.max() + 1
        index = np.argsort(node2graph)
        mapping = np.zeros_like(index)
        mapping[index] = np.arange(len(index))

        node_in, node_out = self.edges
        mask_edge = node2graph[node_in] == node2graph[node_out]
        edge2graph = node2graph[node_in]
        edge_index = index[mask_edge[index]]

        edges = self.edges.copy()
        edges = mapping[edges]

        n_nodes = np.bincount(node2graph, minlength=n_graph)
        n_edges = np.bincount(edge2graph[edge_index], minlength=n_graph)

        cum_nodes = np.cumsum(n_nodes)
        offsets = (cum_nodes - n_nodes)[edge2graph[edge_index]]

        return self.packed_type(edges[:, edge_index], n_nodes=n_nodes, n_edges=n_edges,
                                ndata=self.ndata, edata=self.edata, gdata=self.gdata,
                                n_relation=self.n_relation, offsets=offsets)

    def repeat(self, count: int):
        """_summary_

        Args:
            count (int): _description_

        Returns:
            _type_: _description_
        """
        graphs = [self] * count
        return self.pack(graphs)

    def get_edge(self, edge: np.ndarray):
        """_summary_

        Args:
            edge (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        assert len(edge) == self.edges.shape[1]
        edge_index, _ = self.match(edge)
        return self.edata['weight'][edge_index].sum()

    def match(self, pattern: np.ndarray):
        """_summary_

        Args:
            pattern (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        if pattern:
            index = n_match = np.zeros(0, dtype=int)
            return index, n_match
        if not hasattr(self, 'edge_inverted_index'):
            setattr(self, 'edge_inverted_index', {})
        if pattern.ndim == 1:
            pattern = np.expand_dims(pattern, 0)
        mask = pattern != -1
        scale = 2 ** np.arange(pattern.shape[-1])
        query_type = np.sum(mask * scale, axis=-1)
        query_index = query_type.argsort()
        n_query = np.unique(query_type, return_counts=True)[1]
        query_ends = np.cumsum(n_query, 0)
        query_starts = query_ends - n_query
        mask_set = mask[query_index[query_starts]].tolist()

        type_ranges = []
        type_orders = []
        for i, mask in enumerate(mask_set):
            query_type = tuple(mask)
            type_index = query_index[query_starts[i]: query_ends[i]]
            type_edge = pattern[type_index][:, mask]
            if query_type not in self.edge_inverted_index:
                self.edge_inverted_index[query_type] = self._build_edge_inverted_index(mask)
            inverted_range, order = self.edge_inverted_index[query_type]
            ranges = inverted_range.get(type_edge, default=0)
            type_ranges.append(ranges)
            type_orders.append(order)
        ranges = np.concatenate(type_ranges)
        orders = np.stack(type_orders)
        types = np.arange(len(mask_set))
        types = np.repeat(types, n_query)

        ranges = util.scatter_(ranges, query_index, axis=0, n_axis=len(pattern), reduce='add')
        types = util.scatter_(types, query_index, n_axis=len(pattern), reduce='add')
        starts, ends = ranges.T
        n_match = ends - starts
        offsets = np.cumsum(n_match, 0) - n_match
        types = np.repeat(types, n_match)
        ranges = np.arange(n_match.sum())
        ranges += np.repeat(starts - offsets, n_match)
        index = orders[types, ranges]

        return index, n_match

    def edge2graph(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        idx = np.zeros(self.n_edge, dtype=np.int32)
        if not self.detach:
            idx = ms.Tensor(idx)
        return idx

    def node2graph(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        idx = np.zeros(self.n_node, dtype=np.int32)
        if not self.detach:
            idx = ms.Tensor(idx)
        return idx

    def coord(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if 'coord' in self.ndata:
            coord = self.ndata['coord']
        else:
            coord = None
        return coord

    def degree_in(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return util.scatter_(self.edge_weight, self.edges[0], n_axis=self.n_node, reduce='add')

    def degree_out(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return util.scatter_(self.edge_weight, self.edges[1], n_axis=self.n_node, reduce='add')

    def subgraph(self, index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.mask_node(index, compact=True)

    def compact(self):
        """
        Remove all of isolated nodes for compacting the left connected nod ids.

        Returns:
            Graph
        """
        index = self.degree_out + self.degree_in > 0
        return self.subgraph(index)

    def mask_node(self, index, compact=False):
        """_summary_

        Args:
            index (_type_): _description_
            compact (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        index = self._standarize_index(index, self.n_node)
        mapping = -np.ones(self.n_node, dtype=np.int32)
        if compact:
            mapping[index] = np.arange(len(index))
            n_node = len(index)
        else:
            mapping[index] = index
            n_node = self.n_node
        edges = self.edges.copy()
        edges = mapping[edges]
        edge_index = np.all(edges >= 0, axis=-1)
        ndata = {key: value[index] for key, value in self.ndata.items()}
        edata = {key: value[index] for key, value in self.edata.items()}
        return type(self)(edges[:, edge_index], ndata=ndata, edata=edata, gdata=self.gdata,
                          n_node=n_node, n_relation=self.n_relation)

    def mask_edge(self, index):
        """
        Return the masked graph based on the specified edges.
        This function can also be used to re-order the edges.

        Parameters:
            index (array_like): edge index

        Returns:
            Graph
        """
        index = self._standarize_index(index, self.n_edge)
        edata = {key: value[index] for key, value in self.edata.items()}
        return type(self)(self.edges[:, index], ndata=self.ndata,
                          edata=edata, gdata=self.gdata,
                          n_node=self.n_node, n_relation=self.n_relation)

    def line_graph(self):
        """
        Construct a line graph of this graph.
        The node feature of the line graph is inherited from the edge feature of the original graph.

        In the line graph, each node corresponds to an edge in the original graph.
        For a pair of edges (a, b) and (b, c) that share the same intermediate node in the original graph,
        there is a directed edge (a, b) -> (b, c) in the line graph.

        Returns:
            Graph
        """
        node_in, node_out = self.edges
        edge_index = np.arrange(self.n_edge)
        edge_in = edge_index[np.argsort(node_out)]
        edge_out = edge_index[np.argsort(node_in)]
        degree_in = np.bincount(node_in, minlength=self.n_node)
        degree_out = np.bincount(node_out, minlength=self.n_node)
        size = degree_out * degree_in
        starts = np.repeat(np.cumsum(size, 0) - size, size)
        order = np.arange(np.sum(size))
        local_index = order - starts
        local_inner_size = np.repeat(degree_in, size)
        edge_in_offset = np.repeat(degree_out.cumsum(0) - degree_out, size)
        edge_out_offset = np.repeat(degree_in.cumsum(0) - degree_in, size)
        edge_in_index = local_index // local_inner_size + edge_in_offset
        edge_out_index = local_index % local_inner_size + edge_out_offset

        edge_in = edge_in[edge_in_index]
        edge_out = edge_out[edge_out_index]
        edges = np.stack([edge_in, edge_out])
        ndata = getattr(self, "edata", None)
        return type(self)(edges, n_node=self.n_edge, n_edge=size.sum(),
                          ndata=ndata, gdata=self.gdata)

    def full(self):
        """
        Return a fully connected graph over the nodes.

        Returns:
            Graph
        """
        index = np.arange(self.n_node)
        edges = np.meshgrid(index, index)
        edges = np.stack(edges).reshape(len(edges), -1)
        edata = {'weight': np.ones(len(edges), dtype=np.float32)}
        return type(self)(edges, ndata=self.ndata, edata=edata, gdata=self.gdata,
                          n_node=self.n_node, n_relation=self.n_relation)

    def directed(self, order=None):
        """
        Mask the edges to created a direted graph.
        Edges that go from a node index to a larger or equal node index will be kept.

        Parameters:
            order (array_like): topological order of the nodes
        """
        node_in, node_out = self.edges
        if order is not None:
            edge_index = order[node_in] <= order[node_out]
        else:
            edge_index = node_in <= node_out
        return self.mask_edge(edge_index)

    def undirected(self, add_inverse=False):
        """
        Flip all the edges to create an undirected graph.

        For knowledge graphs, the flipped edges can either have the original relation or an inverse relation.
        The inverse relation for relation :math:`r` is defined as :math:`|R| + r`.

        Parameters:
            add_inverse (bool, optional): whether to use inverse relations for flipped edges
        """
        edges = self.edges.copy()
        edata = self.edata.copy()
        edges = np.flip(edges, axis=1)
        n_relation = self.n_relation
        if n_relation and add_inverse:
            edata['type'] += n_relation
            n_relation *= 2
        edges = np.stack([self.edges, edges], axis=-1)
        edges = util.flatten(edges, 1, 2)
        index = np.expand_dims(np.arange(self.n_edge), -1).repeat(2).flatten()
        edata = {key: value[index] for key, value in self.edata}
        return type(self)(edges, ndata=self.ndata, edata=edata,
                          n_node=self.n_node, n_relation=n_relation)

    def size(self, axis=None):
        """_summary_

        Args:
            axis (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if self.n_relation:
            size = (self.n_node, self.n_node, self.n_relation)
        else:
            size = (self.n_node, self.n_node)
        if axis is None:
            return size
        return size[axis]

    def _check_cache(self, cache: dict):
        if cache is None:
            return {}
        empty_keys = set()
        for key, value in cache.items():
            if value is None:
                empty_keys.add(key)
        for key in empty_keys:
            cache.pop(key)
        return cache

    def _maybe_n_node(self, edges):
        edges = edges.reshape(-1)
        if isinstance(edges, ms.Tensor):
            edges = edges.asnumpy()
        return len(set(edges))

    def _maybe_n_relation(self, edge_type):
        edge_type = edge_type.reshape(-1)
        if isinstance(edge_type, ms.Tensor):
            edge_type = edge_type.asnumpy()
        return len(set(edge_type))

    def _standarize_index(self, index, count):
        """_summary_

        Args:
            index (_type_): _description_
            count (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = np.arange(start, stop, step)
        else:
            index = np.array(index)
        return index

    def _build_edge_inverted_index(self, mask):
        """_summary_

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        keys = self.edges[:, mask]
        base = np.array(self.shape)
        base = base[mask]
        scale = np.cumprod(base, 0)
        scale = scale[-1] // scale
        key = np.sum(keys * scale, axis=-1)
        order = np.argsort(key)
        n_keys = np.unique(key, return_counts=True)[1]
        ends = np.cumsum(n_keys, 0)
        starts = ends - n_keys
        ranges = np.stack([starts, ends], axis=-1)
        key_set = keys[order[starts]]
        # if Dictionaary no problem, use Dictionary(key_set, ranges)
        inverted_range = {tuple(k): tuple(v) for k, v in zip(key_set, ranges)}
        return inverted_range, order


@R.register('data.GraphGatch')
class GraphBatch(Graph):
    """
    Container for sparse graphs with variadic sizes.

    To create a GraphBatch from Graph objects

        >>> batch = data.Graph.pack(graphs)

    To retrieve Graph objects from a GraphBatch

        >>> graphs = batch.unpack()

    .. warning::

        Edges of the same graph are guaranteed to be consecutive in the edge list.
        However, this class doesn't enforce any order on the edges.

    Parameters:
        edges (array_like, optional): list of edges of shape :math:`(|E|, 2)`.
            Each tuple is (node_in, node_out).
        n_nodes (array_like, optional): number of nodes in each graph
            By default, it will be inferred from the largest id in `edges`
        n_edges (array_like, optional): number of edges in each graph
        n_relation (int, optional): number of relations
        ndata (array_like, optional): node feats of shape :math:`(|V|, ...)`
        edata (array_like, optional): edge feats of shape :math:`(|E|, ...)`
        offsets (array_like, optional): node id offsets of shape :math:`(|E|,)`.
        gdata (array_like, optional): grpah features of shape :math:`(|G|,)`
            If not provided, nodes in `edges` should be relative index, i.e., the index in each graph.
            If provided, nodes in `edges` should be absolute index, i.e., the index in the packed graph.
    """

    def __init__(self,
                 edges=None,
                 n_nodes=None,
                 n_edges=None,
                 n_relation=None,
                 cum_nodes=None,
                 cum_edges=None,
                 offsets=None,
                 **kwargs):
        self.n_nodes = n_nodes
        self.n_edges = n_edges

        self.cum_nodes = n_nodes.cumsum(0) if cum_nodes is None else cum_nodes
        self.cum_edges = n_edges.cumsum(0) if cum_edges is None else cum_edges

        self.n_node = n_nodes.sum()
        self.n_edge = n_edges.sum()
        assert self.n_edge == edges.shape[1]

        assert (edges < self.n_node).all()

        if offsets is None:
            offsets = self._get_offsets(self.n_nodes, self.n_edges, self.cum_nodes)
            edges += offsets
        self.offsets = offsets
        super().__init__(edges, n_node=n_nodes.sum(), n_relation=n_relation, **kwargs)

    def __getitem__(self, index):
        if isinstance(index, int):
            item = self.get_item(index)
            return item
        index = self._standarize_index(index, self.batch_size)
        return self.subbatch(index)

    def __len__(self):
        return len(self.n_nodes)

    def __iter__(self):
        setattr(self, '_index', 0)
        return self

    def __next__(self):
        if self._index < self.batch_size:
            item = self[self._index]
            self._index += 1
            return item
        raise StopIteration

    @property
    def batch_size(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return len(self.n_nodes)

    def node2graph(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.detach:
            order = np.arange(self.batch_size)
            idx = order.repeat(self.n_nodes)
        else:
            order = ops.arange(self.batch_size)
            idx = order.repeat(self.n_nodes.asnumpy().tolist())
        return idx

    def edge2graph(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        order = np.arange(self.batch_size)
        idx = order.repeat(self.n_edges)
        if not self.detach:
            idx = ms.Tensor(idx)
        return idx

    def merge(self, graph2graph):
        """_summary_

        Args:
            graph2graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        _, graph2graph = np.unique(graph2graph, return_inverse=True)
        graph_key = graph2graph * self.batch_size + np.arange(self.batch_size)
        graph_index = graph_key.argsort()
        graph = self.subbatch(graph_index)
        graph2graph = graph2graph[graph_index]

        n_graph = graph2graph[-1] + 1
        n_nodes = util.scatter_(graph.n_nodes, graph2graph, n_axis=n_graph, reduce='add')
        n_edges = util.scatter_(graph.n_edges, graph2graph, n_axis=n_graph, reduce='add')
        offsets = self._get_offsets(n_nodes, n_edges)
        graph.n_nodes = n_nodes
        graph.n_edges = n_nodes
        graph.offsets = offsets
        return graph

    def get_item(self, index):
        """
        Get the i-th graph from this packed graph.
        Parameters:
            index (int): graph index
        Returns:
            Graph
        """
        node_index = list(range(self.cum_nodes[index] - self.n_nodes[index], self.cum_nodes[index]))
        edge_index = list(range(self.cum_edges[index] - self.n_edges[index], self.cum_edges[index]))
        edges = self.edges[:, edge_index]
        edges -= self.offsets[edge_index]

        ndata = {key: value[node_index] for key, value in self.ndata.items()}
        edata = {key: value[edge_index] for key, value in self.edata.items()}
        gdata = {key: value[index] for key, value in self.gdata.items()}
        return self.item_type(edges,
                              n_nodes=self.n_nodes[index],
                              ndata=ndata,
                              edata=edata,
                              gdata=gdata,
                              n_relation=self.n_relation)

    def full(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        graphs = self.unpack()
        graphs = [graph.full() for graph in graphs]
        return graphs[0].pack(graphs)

    def unpack(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        graphs = []
        for i in range(self.batch_size):
            graphs.append(self.get_item(i))
        return graphs

    def repeat(self, count):
        """
        Repeat this packed graph. This function behaves similarly to `numpy.repeat`_.

        Parameters:
            count (int): number of repetitions

        Returns:
            GraphBatch
        """
        n_nodes = self.n_nodes.repeat(count)
        n_edges = self.n_edges.repeat(count)
        offsets = self._get_offsets(n_nodes, n_edges)
        edges = self.edges.repeat(count, 1)
        edges += offsets - self.offsets.repeat(count)
        ndata = {key: value.repeat(count) for key, value in self.ndata.item()}
        edata = {key: value.repeat(count) for key, value in self.edata.item()}
        gdata = {key: value.repeat(count) for key, value in self.gdata.item()}
        return type(self)(edges, ndata=ndata, edata=edata, gdata=gdata,
                          n_nodes=n_nodes, n_edges=n_edges, n_relation=self.n_relation,
                          offsets=offsets)

    def repeat_interleave(self, repeats):
        """
        Repeat this packed graph. This function behaves similarly to `torch.repeat_interleave`_.

        .. _torch.repeat_interleave:
            https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html

        Parameters:
            repeats (Tensor or int): number of repetitions for each graph

        Returns:
            PackedGraph
        """
        if np.prod(repeats.shape) == 1:
            repeats = repeats * np.ones(self.batch_size)
        n_nodes = self.n_nodes.repeat(repeats)
        n_edges = self.n_edges.repeat(repeats)
        cum_nodes = n_nodes.cumsum(0)
        cum_edges = n_edges.cumsum(0)
        n_node = n_nodes.sum()
        n_edge = n_edges.sum()
        batch_size = repeats.sum()

        # special case 1: graphs[i] may have no node or no edge
        # special case 2: repeats[i] may be 0
        cum_repeats_shifted = repeats.cumsum(0) - repeats
        mask_graph = cum_repeats_shifted < batch_size
        cum_repeats_shifted = cum_repeats_shifted[mask_graph]

        index = cum_nodes - n_nodes
        index = np.concatenate([index, index[cum_repeats_shifted]])
        value = np.concatenate([-n_nodes, self.n_nodes[mask_graph]])
        mask = index < n_node
        node_index = util.scatter_add(value[mask], index[mask], n_axis=n_node)
        node_index = (node_index + 1).cumsum(0) - 1

        index = cum_edges - n_edges
        index = np.concatenate([index, index[cum_repeats_shifted]])
        value = np.concatenate([-n_edges, self.n_edges[mask_graph]])
        mask = index < n_edge
        edge_index = util.scatter_add(value[mask], index[mask], n_axis=n_edge)
        edge_index = (edge_index + 1).cumsum(0) - 1

        graph_index = np.arange(self.batch_size).repeat(repeats)

        offsets = self._get_offsets(n_nodes, n_edges)
        edges = self.edges[:, edge_index]
        edges += offsets - self.offsets[edge_index]

        ndata = {key: value[node_index] for key, value in self.ndata.item()}
        edata = {key: value[edge_index] for key, value in self.edata.item()}
        gdata = {key: value[graph_index] for key, value in self.gdata.item()}
        return type(self)(edges, ndata=ndata, edata=edata, gdata=gdata,
                          n_nodes=n_nodes, n_edges=n_edges, n_relation=self.n_relation,
                          offsets=offsets)

    def mask_node(self, index, compact=False):
        """
        Return a masked packed graph based on the specified nodes.

        Note the compact option is only applied to node ids but not graph ids.
        To generate compact graph ids, use :meth:`subbatch`.

        Parameters:
            index (array_like): node index
            compact (bool, optional): compact node ids or not

        Returns:
            GraphBatch
        """
        index = self._standarize_index(index, self.n_node)
        mapping = -np.ones(self.n_node, dtype=np.int32)
        if compact:
            mapping[index] = np.arange(len(index))
            n_nodes = self._masked_n_xs(index, self.cum_nodes)
            offsets = self._get_offsets(n_nodes, self.n_edges)
        else:
            mapping[index] = index
            n_nodes = self.n_nodes
            offsets = self.offsets

        edges = self.edges.clone()
        edges = mapping[edges]
        node_index = mapping >= 0
        edge_index = (edges >= 0).all(axis=-1)
        n_edges = self._masked_n_xs(edge_index, self.cum_edges)
        ndata = {key: value[node_index] for key, value in self.ndata.item()}
        edata = {key: value[edge_index] for key, value in self.edata.item()}
        return type(self)(edges[edge_index], n_nodes=n_nodes, n_edges=n_edges,
                          ndata=ndata, edata=edata, gdata=self.gdata,
                          n_relation=self.n_relation, offsets=offsets[edge_index])

    def mask_edge(self, index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        index = self._standarize_index(index, self.n_edges)
        n_edges = self._masked_n_xs(index, self.cum_edges)
        edata = {key: value[index] for key, value in self.edata}
        return type(self)(self.edges[:, index], n_nodes=self.n_nodes, n_edges=n_edges,
                          ndata=self.ndata, edata=edata, gdata=self.gdata,
                          n_relation=self.n_relation, offsets=self.offsets[index])

    def mask_graph(self, index, compact=False):
        """_summary_

        Args:
            index (_type_): _description_
            compact (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        index = self._standarize_index(index, self.batch_size)
        graph_mapping = np.ones(self.batch_size, dtype=np.int32)
        graph_mapping[index] = np.arange(len(index))
        node_index = graph_mapping[self.node2graph()] >= 0
        node_index = self._standarize_index(node_index, self.n_node)
        graph_index = graph_mapping >= 0
        mapping = -np.ones(self.n_node, dtype=np.int32)

        if compact:
            key = graph_mapping[self.node2graph()[node_index]] * self.n_node + node_index
            order = np.argsort(key)
            node_index = node_index[order]
            mapping[node_index] = np.arange(len(node_index))
            n_nodes = self.n_nodes[index]
        else:
            mapping[node_index] = node_index
            n_nodes = np.zeros_like(self.n_nodes)
            n_nodes[index] = self.n_nodes[index]

        edges = self.edges.clone()
        edges = mapping[edges]
        edge_index = (edges >= 0).all(dim=-1)
        edge_index = self._standarize_index(edge_index, self.n_edges)
        if compact:
            key = graph_mapping[self.edge2graph()[edge_index]] * self.n_edges + edge_index
            order = np.argsort(key)
            edge_index = edge_index[order]
            n_edges = self.n_edges[index]
        else:
            n_edges = np.zeros_like(self.n_edges)
            n_edges[index] = self.n_edges[index]
        offsets = self._get_offsets(n_nodes, n_edges)
        ndata = {key: value[node_index] for key, value in self.ndata.item()}
        edata = {key: value[edge_index] for key, value in self.edata.item()}
        gdata = {key: value[graph_index] for key, value in self.gdata.item()}
        return type(self)(edges[:, edge_index], n_nodes=n_nodes, n_edges=n_edges,
                          ndata=ndata, edata=edata, gdata=gdata,
                          n_relation=self.n_relation, offsets=offsets)

    def subbatch(self, index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.mask_graph(index, compact=True)

    def line_graph(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        node_in, node_out = self.edges
        edge_index = np.arange(self.n_edges)
        edge_in = edge_index[node_out.argsort()]
        edge_out = edge_index[node_in.argsort()]

        degree_in = node_in.bicount(minlength=self.n_node)
        degree_out = node_out.bincount(minlength=self.n_node)
        size = degree_in * degree_out
        starts = np.repeat(np.cumsum(size, 0) - size, size)
        order = np.arange(np.sum(size))

        local_index = order - starts
        local_inner_size = np.repeat(degree_in, size)
        edge_in_offset = np.repeat(np.cumsum(degree_out, 0) - degree_out, size)
        edge_out_offset = np.repeat(np.cumsum(degree_in, 0) - degree_in, size)
        edge_in_index = local_index // local_inner_size + edge_in_offset
        edge_out_index = local_index % local_inner_size + edge_out_offset

        edge_in = edge_in[edge_in_index]
        edge_out = edge_out[edge_out_index]
        edges = np.stack([edge_in, edge_out])
        ndata = getattr(self, 'edata', None)
        n_nodes = self.n_edges
        n_edges = util.scatter_add(size, self.node2graph(), axis=0, n_axis=self.batch_size)
        offsets = self._get_offsets(n_nodes, n_edges)
        return type(self)(edges, n_nodes=n_nodes, n_edges=n_edges, offsets=offsets, ndata=ndata)

    def undirected(self, add_inverse=False):
        """
        Flip all the edges to create undirected graphs.

        For knowledge graphs, the flipped edges can either have the original relation or an inverse relation.
        The inverse relation for relation :math:`r` is defined as :math:`|R| + r`.

        Parameters:
            add_inverse (bool, optional): whether to use inverse relations for flipped edges
        """
        edges = self.edges.clone()
        edata = self.edata.clone()
        edges = np.flip(edges, 1)
        n_relation = self.n_relation
        if n_relation and add_inverse:
            edata['type'] += n_relation
            n_relation = n_relation * 2
        edges = util.flatten(np.stack([self.edges, edges], axis=-1), 1, 2)
        offsets = self.offsets.expand_dims(-1).repeat(2).flatten()
        index = np.arange(self.n_edges).expand_dims(-1)
        index = np.repeat(index, 2).flatten()
        edata = {key: value[index] for key, value in self.edata.items()}
        return type(self)(edges, n_nodes=self.n_nodes, n_edges=self.n_edges * 2,
                          ndata=self.ndata, edata=edata, gdata=self.gdata,
                          n_relation=n_relation, offsets=offsets)

    def _get_offsets(self, n_nodes=None, n_edges=None, cum_nodes=None, cum_edges=None):
        """_summary_

        Args:
            n_nodes (_type_, optional): _description_. Defaults to None.
            n_edges (_type_, optional): _description_. Defaults to None.
            cum_nodes (_type_, optional): _description_. Defaults to None.
            cum_edges (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if n_nodes is None:
            n_nodes = np.diff(cum_nodes, prepend=0)
        if n_edges is None:
            n_edges = np.diff(cum_edges, prepend=0)
        if cum_nodes is None:
            cum_nodes = np.cumsum(n_nodes, 0)
        return np.repeat(cum_nodes - n_nodes, n_edges)

    def _masked_n_xs(self, mask, cum_xs):
        """_summary_

        Args:
            mask (_type_): _description_
            cum_xs (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = np.zeros(cum_xs[-1], dtype=int)
        x[mask] = 1
        cum_indexes = x.cumsum(0)
        cum_indexes = np.concatenate([np.zeros(1, dtype=int), cum_indexes])
        new_cum_xs = cum_indexes[cum_xs]
        prepend = np.zeros(1, dtype=int)
        new_num_xs = np.diff(new_cum_xs, prepend=prepend)
        return new_num_xs


Graph.batch_type = GraphBatch
GraphBatch.item_type = Graph
