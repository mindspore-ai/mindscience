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
data class of Graph and Graphbatch
"""
from typing import Union
from copy import deepcopy
from dataclasses import dataclass
import mindspore as ms
from mindspore import ops
import numpy as np
from scipy import spatial

from .. import core
from .. import utils
from ..features import mol_feat as feature
from ..configs import Registry as R


@R.register('data.Graph')
@ms.jit_class
@dataclass
class Graph(core.MetaData):
    """ The data structure of Graph. Each graph contains N nodes and M edges.

    Args:
        edges (np.ndarray, ms.Tensor):      The 2 x M matrix of edges. The first row is index of start nodes
                                            and the second row is the index of end nodes.
        n_node (int):                       Number of nodes.
        n_relation (int):                   Number of different types of edges.
        node_type (np.ndarray, ms.Tensor):  The matrix of edge types. The shape is (M, ).
        node_feat (np.ndarray, ms.Tensor):  The matrix of node features. The shape is (N, Fn).
        edge_type (np.ndarray, ms.Tensor):  The matrix of edge types. The shape is (M, ).
        edge_feat (np.ndarray, ms.Tensor):  The matrix of edge features. The shape is (M, Fm).
        graph_feat (np.ndarray, ms.Tensor): The matrix of graph features. The shape is (Fg, ).
        node_coord: (np.ndarray, ms.Tensor): The matrix of node coordinates. The shape is (N, 3).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    edges: np.ndarray = None
    n_node: int = None
    n_relation: int = None
    node_type: np.ndarray = None
    node_feat: np.ndarray = None
    edge_type: np.ndarray = None
    edge_weight: np.ndarray = None
    edge_feat: np.ndarray = None
    graph_feat: np.ndarray = None
    node_coord: np.ndarray = None

    def __post_init__(self):
        op_zeros = np.zeros if self.detach else ops.zeros
        if self.n_node is None:
            self.n_node = self._maybe_n_node(self.edges)
        if self.n_relation is None:
            self.n_relation = self._maybe_n_relation(self.edge_type)
        if self.node_type is None:
            self.node_type = op_zeros(self.n_node, dtype=int)
        if self.edge_type is None:
            self.edge_type = op_zeros(self.n_edge, dtype=int)
        if self.edge_weight is None:
            self.edge_weight = op_zeros(self.n_edge)
        super().__post_init__()

    def __getitem__(self, index):
        # why do we check tuple?
        # case 1: x[0, 1] is parsed as (0, 1)
        # case 2: x[[0, 1]] is parsed as [0, 1]
        if not isinstance(index, tuple):
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
        return self.size()

    @property
    def coord(self):
        return self.node_coord

    @property
    def batch_size(self):
        return 1

    @property
    def n_edge(self):
        if self.edges is None or 0 in self.edges.shape:
            return 0
        return self.edges.shape[1]

    @property
    def node2graph(self):
        """
        The matrix contains index to indicate which graph the node belongs to.

        Returns:
            idx (ms.Tensor, np.ndarray): index matrix.
        """
        idx = np.zeros(self.n_node, dtype=np.int32)
        if not self.detach:
            idx = ms.Tensor(idx)
        return idx

    @property
    def edge2graph(self):
        """
        The matrix contains index to indicate which graph the edge belongs to.

        Returns:
            idx (ms.Tensor, np.ndarray): index matrix.
        """
        idx = np.zeros(self.n_edge, dtype=np.int32)
        if not self.detach:
            idx = ms.Tensor(idx)
        return idx

    @classmethod
    def from_dense(cls, adjacency: Union[np.ndarray, ms.Tensor], **kwargs):
        """
        Construct the graph object based on the adjcency matrix

        Args:
            adjacency (np.ndarray): adjcency matrix

        Returns:
           Graph
        """
        assert adjacency.shape[0] == adjacency.shape[1]
        edges = adjacency.nonzero().T
        kwargs['n_node'] = adjacency.shape[0]
        kwargs['n_relation'] = adjacency.shape[2] if adjacency.ndim > 2 else None
        if 'edge_weight' not in kwargs:
            kwargs['edge_weight'] = adjacency[adjacency != 0]
        return cls(edges=edges, **kwargs)

    @classmethod
    def knn_graph(cls, coord: np.ndarray, max_neighbors: int = None, cutoff: float = None, **kwargs):
        """K-Nearest Neighbor methods to construct graph data from coordinate. The distance metric is Eucidean distance.

        Args:
            coord (np.ndarray): The coordinate matrix that contains the spatial position each node. Shape: N x 3
            max_neighbors (int, optional): The maximum number of each node. Defaults to None.
            cutoff (float, optional): The maximum distance of allowable edge between two nodes. Defaults to None.

        Returns:
            Graph
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
        edge_feat = feature.distance(dist_list=dist_list, **kwargs)
        graph = cls(edges=edges.T, n_node=dist.shape[0], edge_feat=edge_feat, node_coord=coord)
        return graph

    @classmethod
    def pack(cls, graphs: list):
        """
        Pack a list of Graph objects into a GraphBatch object.

        Args:
            graphs (list): a list of graphs. For each property, the value from all of graphs must have same shape.

        Returns:
            graph (GraphBatch): Packed GraphBatch object.
        """
        kwargs = {'n_relation': graphs[0].n_relation}
        detach = all([isinstance(g.cls_name, str) for g in graphs])
        op_array = np.array if detach else ms.Tensor
        op_concat = np.concatenate if detach else ops.concat
        for key in cls.batch_type.keys():
            if key in ['cls_name', 'detach', 'n_node', 'n_edge', 'offsets', 'cum_nodes', 'cum_edges']:
                continue
            if key == 'n_nodes':
                values = [getattr(g, 'n_node') for g in graphs]
            elif key == 'n_edges':
                values = [g.n_edge if g.edges is not None else 0 for g in graphs]
            else:
                values = [getattr(g, key) for g in graphs]
            values = [v for v in values if v is not None]
            if key == 'edges' and values:
                kwargs[key] = op_concat(values, axis=1)
            elif key == 'n_relation':
                assert (op_array(values) == values[0]).all()
                kwargs[key] = values[0]
            elif not values or isinstance(values[0], (int, float, bool, np.number)):
                kwargs[key] = op_array(values)
            else:
                kwargs[key] = op_concat(values)
        graph = cls.batch_type(**kwargs)
        return graph

    def connected_components(self):
        """
        Find the subgraphs that no edge connects each pair the them.

        Returns:
            graphs (GraphBatch): The graphs of connected components
            n_nodes (List[int]): The n_node of each connected components
        """
        node_in, node_out = self.edges
        order = np.arange(self.n_node)
        node_in = np.concatenate([node_in, node_out, order])
        node_out = np.concatenate([node_out, node_in, order])

        min_neighbor = np.arange(self.n_node)
        last = np.zeros_like(min_neighbor)
        while not np.equal(min_neighbor, last):
            last = min_neighbor
            min_neighbor = utils.scatter_min(min_neighbor[node_out], node_in, n_axis=self.n_node)
        anchor = np.unique(min_neighbor)
        n_nodes = self.node2graph[anchor].bincount(minlength=self.batch_size)
        graphs = self.split(min_neighbor)
        return graphs, n_nodes

    def split(self, node2graph: np.ndarray):
        """
        Split a graph into a list of graphs.

        Args:
            node2graph (np.ndarray): The matrix contains index to indicate which graph the edge belongs to.

        Returns:
            graphs (List[Graph]): a list of graphs split from itself.
        """
        _, node2graph = np.unique(node2graph, return_inverse=True)
        n_graph = node2graph.max() + 1
        index = node2graph.float().argsort()
        mapping = np.zeros_like(index)
        mapping[index] = np.arange(len(index))

        node_in, node_out = self.edges
        mask_edge = node2graph[node_in] == node2graph[node_out]
        edge2graph = node2graph[node_in]
        edge_index = edge2graph.float().argsort()
        edge_index = index[mask_edge[edge_index]]

        is_first_node = np.diff(node2graph[index], prepend=-1)
        graph_index = self.node2graph[index[is_first_node]]
        kwargs = deepcopy(self.__dict__)
        edges = self.edges.copy()
        edges = mapping[edges]
        kwargs['edges'] = edges[:, edge_index]
        kwargs['n_nodes'] = np.bincount(node2graph, minlength=n_graph)
        kwargs['n_edges'] = np.bincount(edge2graph[edge_index], minlength=n_graph)

        cum_nodes = kwargs['n_nodes'].cumsum(0)
        kwargs['offsets'] = (cum_nodes - kwargs['n_nodes'])[edge2graph[edge_index]]
        kwargs.update(self.data_mask(node_index=index, edge_index=edge_index, graph_index=graph_index))
        graphs = self.packed_type(**kwargs)
        return graphs

    def repeat(self, repeats):
        """

        Packed a number of same graph into a GraphBatch object.

        Args:
            count (int): the number of repeat

        Returns:
            graph (GraphBatch): a GraphBatch object after repeating itself `count` times
        """
        graphs = [self] * repeats
        graph = self.pack(graphs)
        return graph

    def get_edge(self, edge: np.ndarray):
        """
        Get the total weight of given edges.

        Args:
            edge (np.ndarray): a list of edges.

        Returns:
            weight (float): Total of matched edges.
        """
        assert len(edge) == self.n_edge
        edge_index, _ = self.match(edge)
        weight = self.edge_weight[edge_index].sum()
        return weight

    def match(self, pattern: np.ndarray):
        """
        Searching the matched part of graph by given pattern

        Args:
            pattern (np.ndarray): a list of edges

        Returns:
            index (np.ndarray): The index of edge that matches the pattern
            n_match (int): The count of matched subgraph
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
        query_index = query_type.float().argsort()
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

        ranges = utils.scatter_(ranges, query_index, axis=0, n_axis=len(pattern), reduce='add')
        types = utils.scatter_(types, query_index, n_axis=len(pattern), reduce='add')
        starts, ends = ranges.T
        n_match = ends - starts
        offsets = np.cumsum(n_match, 0) - n_match
        types = np.repeat(types, n_match)
        ranges = np.arange(n_match.sum())
        ranges += np.repeat(starts - offsets, n_match)
        index = orders[types, ranges]
        return index, n_match

    def degree_in(self):
        """
        Total weight of edges for each node as source.

        Returns:
            degree_in (np.ndarray, ms.Tensor): The degree input of each node.
        """
        degree_in = utils.scatter_(self.edge_weight, self.edges[0], n_axis=self.n_node, reduce='add')
        return degree_in

    def degree_out(self):
        """
        Total weight of edges for each node as target.

        Returns:
            degree_out (np.ndarray, ms.Tensor): The degree output of each node.
        """
        return utils.scatter_(self.edge_weight, self.edges[1], n_axis=self.n_node, reduce='add')

    def subgraph(self, index):
        """
        Generating Subgraph based on given index of nodes.

        Args:
            index (np.ndarray, ms.Tensor): index of nodes

        Returns:
            graph (Graph): subgraph that contain the node indicated in the index.
        """
        graph = self.mask_node(index, compact=True)
        return graph

    def compact(self):
        """
        Remove all of isolated nodes for compacting the left connected nod ids.

        Returns:
            Graph
        """
        index = self.degree_out + self.degree_in > 0
        return self.subgraph(index)

    def data_mask(self, node_index=None, edge_index=None, graph_index=None):
        """
        selecting property data whose name starts with ``node_``, ``edge_`` or ``graph_`` with mask.
        Args:
            node_index (int, slice, array_like, optional): node index. Defaults to None.
            edge_index (int, slice, array_like, optional): edge index. Defaults to None.
            graph_index (int, slice, array_like, optional): graph index. Defaults to None.

        Returns:
            outputs (dict): masked property data whose name starts with ``node_``, ``edge_`` or ``graph_``.
        """
        kwargs = deepcopy(self.__dict__)
        outputs = {}
        for key, value in kwargs.items():
            if value is None:
                continue
            if node_index is not None and key.startswith('node_'):
                outputs[key] = value[node_index]
            if edge_index is not None and key.startswith('edge_'):
                outputs[key] = value[edge_index]
            if graph_index is not None and key.startswith('graph_'):
                outputs[key] = value[graph_index]
        return outputs

    def mask_node(self, index, compact=False):
        """
        Remove those nodes which not in the given index.

        Args:
            index (int, slice, array_like): The node that need to be not removed.
            compact (bool, optional): If True, the unmasked node will be reindexing. Defaults to False.

        Returns:
            Graph: masked Graph object.
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
        kwargs = deepcopy(self.__dict__)
        kwargs['edges'] = self.edges[:, edge_index]
        kwargs['n_node'] = n_node
        kwargs.update(self.data_mask(node_index=index, edge_index=edge_index))
        return self.from_dict(**kwargs)

    def mask_edge(self, index):
        """
        Return the masked graph based on the specified edges.
        This function can also be used to re-order the edges.

        args:
            index (int, slice, array_like): edge index

        Returns:
            Graph: the Graph object after removing masked edges.
        """
        index = self._standarize_index(index, self.n_edge)
        kwargs = deepcopy(self.__dict__)
        kwargs['edges'] = self.edges[:, index]
        kwargs.update(self.data_mask(edge_index=index))
        return self.from_dict(**kwargs)

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
        starts = np.repeat(size.cumsum(0) - size, size)
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
        return type(self)(edges=edges, n_node=self.n_edge,
                          node_type=self.edge_type, node_feat=self.edge_feat,
                          graph_feat=self.graph_feat)

    def full(self):
        """
        Return a fully connected graph over the nodes.

        Returns:
            Graph
        """
        kwargs = deepcopy(self.__dict__)
        index = np.arange(self.n_node)
        edges = np.meshgrid(index, index)
        edges = np.stack(edges).reshape(len(edges), -1)
        edge_weight = np.ones(len(edges), dtype=np.float32)
        kwargs['edges'] = edges
        kwargs['edge_weight'] = edge_weight
        return self.from_dict(**kwargs)

    def directed(self, order=None):
        """
        Mask the edges to created a direted graph.
        Edges that go from a node index to a larger or equal node index will be kept.

        Args:
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

        Args:
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
        edges = utils.flatten(edges, 1, 2)
        index = np.expand_dims(np.arange(self.n_edge), -1).repeat(2).flatten()
        kwargs = deepcopy(self.__dict__)
        kwargs['edges'] = edges
        kwargs = {key: value[index] if key.startswith('edge_') else value for key, value in kwargs.items()}
        return self.from_dict(**kwargs)

    def size(self, axis: Union[int, tuple, list] = None):
        """The shape of the graph

        Args:
            axis (int, tuple, list, optional): the axis of sizes. Defaults to None.

        Returns:
            size (int, tuple, list): the shape with given axis.
        """
        if self.n_relation:
            size = (self.n_node, self.n_node, self.n_relation)
        else:
            size = (self.n_node, self.n_node)
        if axis is None:
            return size
        size = size[axis]
        return size

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
        """
        Standardize the index

        Args:
            index (int, list, tuple, slice, np.ndarray, ms.Tensor): index
            count (int): if index is slice and the end is not given, count will be viewed as the end.

        Returns:
            index (list): Standardized index
        """
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = np.arange(start, stop, step, dtype=np.int32)
            if not self.detach:
                index = ms.Tensor.from_numpy(index)
        elif isinstance(index, (np.ndarray, ms.Tensor)):
            if index.ndim == 0:
                index = index.expand_dims(0)
            if index.dtype in [np.bool_, ms.bool_]:
                if index.shape != (count,):
                    raise IndexError("Invalid mask. Expect mask to have shape %s, but found %s" %
                                     ((int(count),), tuple(index.shape)))
                index = index.nonzero().squeeze(-1)
            else:
                max_index = -1 if index.shape[0] == 0 else index.max()
                if max_index >= count:
                    raise IndexError("Invalid index. Expect index smaller than %d, but found %d" % (count, max_index))
        return index

    def _build_edge_inverted_index(self, mask):
        """
        Build inverted edge index.

        Args:
            mask (int, list, tuple, slice, np.ndarray, ms.Tensor): index of edges need to be masked

        Returns:
            inverted_range (dict): inverted range of edges
            order (list): the order of edges
        """
        keys = self.edges[:, mask]
        base = np.array(self.shape)
        base = base[mask]
        scale = np.cumprod(base, 0)
        scale = scale[-1] // scale
        key = np.sum(keys * scale, axis=-1)
        order = np.argsort(key)
        n_keys = np.unique(key, return_counts=True)[1]
        ends = n_keys.cumsum(0)
        starts = ends - n_keys
        ranges = np.stack([starts, ends], axis=-1)
        key_set = keys[order[starts]]
        # if Dictionaary no problem, use Dictionary(key_set, ranges)
        inverted_range = {tuple(k): tuple(v) for k, v in zip(key_set, ranges)}
        return inverted_range, order


@R.register('data.GraphGatch')
@dataclass
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

    Args:
        n_nodes (array_like, optional): number of nodes in each graph
            By default, it will be inferred from the largest id in `edges`
        n_edges (array_like, optional): number of edges in each graph
        n_relation (int, optional): number of relations
        offsets (array_like, optional): node id offsets of shape :math:`(|E|,)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    n_nodes: np.ndarray = None
    n_edges: np.ndarray = None
    offsets: np.ndarray = None
    cum_nodes: np.ndarray = None
    cum_edges: np.ndarray = None

    def __post_init__(self):
        self.n_node = int(self.n_nodes.sum())
        self.cum_nodes = self.n_nodes.cumsum(0)
        self.cum_edges = self.n_edges.cumsum(0)
        if self.offsets is None:
            repeats = self.n_edges if self.detach else self.n_edges.asnumpy().tolist()
            self.offsets = (self.n_nodes.cumsum(0) - self.n_nodes).repeat(repeats)
            self.edges += self.offsets
        super().__post_init__()

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            item = self.get_item(int(index))
            return item
        index = self._standarize_index(index, self.batch_size)
        return self.subbatch(index)

    def __len__(self):
        return len(self.n_nodes)

    @property
    def batch_size(self):
        """The number of graphs it contains"""
        return len(self.n_nodes)

    @property
    def node2graph(self):
        """
        The matrix contains index to indicate which graph the node belongs to.

        Returns:
            idx (ms.Tensor, np.ndarray): index matrix.
        """
        if self.detach:
            order = np.arange(self.batch_size, dtype=np.int32)
            idx = order.repeat(self.n_nodes)
        else:
            order = ops.arange(self.batch_size).int()
            idx = order.repeat(self.n_nodes.asnumpy().tolist())
        return idx

    @property
    def edge2graph(self):
        """
        The matrix contains index to indicate which graph the edge belongs to.

        Returns:
            idx (ms.Tensor, np.ndarray): index matrix.
        """
        if not self.detach:
            n_edges = self.n_edges.asnumpy().tolist()
            order = ops.arange(self.batch_size).int()
        else:
            n_edges = self.n_edges
            order = np.arange(self.batch_size, dtype=np.int32)
        idx = order.repeat(n_edges)
        return idx

    def merge(self, graph2graph):
        """
        Merge the graphs it contains based on the indics in graph2graph

        Args:
            graph2graph (np.ndarray): indics that which graph to merge

        Returns:
            graph (GraphBatch): Merged GraphBatch object
        """
        _, graph2graph = np.unique(graph2graph, return_inverse=True)
        graph_key = graph2graph * self.batch_size + np.arange(self.batch_size)
        graph_index = graph_key.float().argsort()
        graph = self.subbatch(graph_index)
        graph2graph = graph2graph[graph_index]

        n_graph = graph2graph[-1] + 1
        n_nodes = utils.scatter_(graph.n_nodes, graph2graph, n_axis=n_graph, reduce='add')
        n_edges = utils.scatter_(graph.n_edges, graph2graph, n_axis=n_graph, reduce='add')
        offsets = self._get_offsets(n_nodes, n_edges)
        graph.n_nodes = n_nodes
        graph.n_edges = n_nodes
        graph.offsets = offsets
        return graph

    def get_item(self, index):
        """
        Get the i-th graph from this packed graph.

        Args:
            index (int): graph index

        Returns:
            Graph
        """
        node_index = list(range(self.cum_nodes[index] - self.n_nodes[index], self.cum_nodes[index]))
        edge_index = list(range(self.cum_edges[index] - self.n_edges[index], self.cum_edges[index]))

        kwargs = deepcopy(self.__dict__)
        edges = self.edges[:, edge_index]
        edges -= self.offsets[edge_index]

        kwargs['edges'] = edges
        kwargs['n_node'] = self.n_nodes[index]
        kwargs.update(self.data_mask(node_index=node_index, edge_index=edge_index, graph_index=index))

        return self.build(self.item_type, **kwargs)

    def full(self):
        """
        Transform each graph it contains to the fully connected graph over the nodes.

        Returns:
            GraphBatch: A GraphBatch object that contains all of fully connected graph.
        """
        graphs = self.unpack()
        graphs = [graph.full() for graph in graphs]
        return graphs[0].pack(graphs)

    def unpack(self):
        """Unpack a GraphBatch object into a list graphs.

        Returns:
            graphs (list): a list of unpacked graph.
        """
        graphs = []
        for i in range(self.batch_size):
            graphs.append(self.get_item(i))
        return graphs

    def tile(self, count):
        """
        Repeat this packed graph. This function behaves similarly to `numpy.tile`_.

        Args:
            count (int): number of repetitions

        Returns:
            GraphBatch
        """
        kwargs = deepcopy(self.__dict__)
        n_nodes = self.n_nodes.tile((count,))
        n_edges = self.n_edges.tile((count,))
        offsets = self._get_offsets(n_nodes, n_edges)
        edges = self.edges.tile((1, count))
        edges += offsets - self.offsets.tile((count,))
        kwargs['edges'] = edges
        kwargs['n_nodes'] = n_nodes
        kwargs['n_edges'] = n_edges
        kwargs['offsets'] = offsets
        for key, value in kwargs.items():
            if key[:5] in ('node_', 'edge_', 'graph') and value is not None:
                shape = [count] + [1] * (value.ndim - 1)
                value = value.tile(tuple(shape))
                kwargs[key] = value

        return self.build(type(self), **kwargs)

    def repeat(self, repeats):
        """
        Repeat this packed graph. This function behaves similarly to `numpy.repeat`_.

        Args:
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
        node_index = utils.scatter_add(value[mask], index[mask], n_axis=n_node)
        node_index = (node_index + 1).cumsum(0) - 1

        index = cum_edges - n_edges
        index = np.concatenate([index, index[cum_repeats_shifted]])
        value = np.concatenate([-n_edges, self.n_edges[mask_graph]])
        mask = index < n_edge
        edge_index = utils.scatter_add(value[mask], index[mask], n_axis=n_edge)
        edge_index = (edge_index + 1).cumsum(0) - 1

        graph_index = np.arange(self.batch_size).repeat(repeats)

        offsets = self._get_offsets(n_nodes, n_edges)
        edges = self.edges[:, edge_index]
        edges += offsets - self.offsets[edge_index]

        kwargs = deepcopy(self.__dict__)
        kwargs['edges'] = edges
        kwargs['n_nodes'] = n_nodes
        kwargs['n_edges'] = n_edges
        kwargs['offsets'] = offsets
        kwargs.update(self.data_mask(node_index=node_index, edge_index=edge_index, graph_index=graph_index))
        return self.build(type(self), **kwargs)

    def mask_node(self, index, compact=False):
        """
        Return a masked packed graph based on the specified nodes.

        Note the compact option is only applied to node ids but not graph ids.
        To generate compact graph ids, use :meth:`subbatch`.

        Args:
            index (array_like): node index
            compact (bool, optional): compact node ids or not

        Returns:
            GraphBatch
        """
        op_ones = np.ones if self.detach else ops.ones
        op_range = np.arange if self.detach else ops.arange
        int32 = np.int32 if self.detach else ms.int32
        index = self._standarize_index(index, self.n_node)
        mapping = -op_ones(self.n_node, dtype=int32)
        if compact:
            mapping[index] = op_range(len(index))
            n_nodes = self._masked_n_xs(index, self.cum_nodes)
            offsets = self._get_offsets(n_nodes, self.n_edges)
        else:
            mapping[index] = index
            n_nodes = self.n_nodes
            offsets = self.offsets

        edges = self.edges.copy()
        edges = mapping[edges]
        edge_index = (edges >= 0).all(axis=0)
        n_edges = self._masked_n_xs(edge_index, self.cum_edges)

        kwargs = deepcopy(self.__dict__)
        kwargs['edges'] = edges[:, edge_index]
        kwargs['n_nodes'] = n_nodes
        kwargs['n_edges'] = n_edges
        kwargs['offsets'] = offsets[edge_index]
        if compact:
            kwargs.update(self.data_mask(node_index=index, edge_index=edge_index))
        else:
            kwargs.update(self.data_mask(edge_index=edge_index))
        return self.build(type(self), **kwargs)

    def mask_edge(self, index):
        """
        Return a masked packed graph based on the specified edges.

        Args:
            index (array_like): edge index

        Returns:
            GraphBatch
        """
        index = self._standarize_index(index, self.n_edge)
        n_edges = self._masked_n_xs(index, self.cum_edges)
        kwargs = deepcopy(self.__dict__)
        kwargs['edges'] = kwargs.get('edges')[:, index]
        kwargs['n_edges'] = n_edges
        kwargs['offsets'] = self.offsets[index]
        kwargs.update(self.data_mask(edge_index=index))
        return self.build(type(self), **kwargs)

    def mask_graph(self, index, compact=False):
        """
        Return a masked packed graph based on the specified graphs.
        Args:
            index (array_like): grpah index
            compact (bool, optional): compact node ids or not. Defaults to False.

        Returns:
            GraphBatch
        """
        op_ones = np.ones if self.detach else ops.ones
        op_zeros = np.zeros if self.detach else ops.zeros
        op_range = np.arange if self.detach else ops.arange
        int32 = np.int32 if self.detach else ms.int32

        index = self._standarize_index(index, self.batch_size)
        if index.shape[0] == 0:
            return None
        graph_mapping = -op_ones(self.batch_size, dtype=int32)
        graph_mapping[index] = op_range(len(index))

        node_index = graph_mapping[self.node2graph] >= 0
        node_index = self._standarize_index(node_index, self.n_node)
        mapping = -op_ones(self.n_node, dtype=int32)

        if compact:
            key = graph_mapping[self.node2graph[node_index]] * self.n_node + node_index
            order = key.float().argsort()
            node_index = node_index[order]
            mapping[node_index] = op_range(len(node_index))
            n_nodes = self.n_nodes[index]
        else:
            mapping[node_index] = node_index
            n_nodes = op_zeros(self.n_nodes.shape)
            n_nodes[index] = self.n_nodes[index]

        edges = self.edges.copy()
        edges = mapping[edges]
        edge_index = (edges >= 0).all(axis=0)
        edge_index = self._standarize_index(edge_index, self.n_edge)
        if compact:
            key = graph_mapping[self.edge2graph[edge_index]] * self.n_edge + edge_index
            order = key.float().argsort()
            edge_index = edge_index[order]
            n_edges = self.n_edges[index]
        else:
            n_edges = op_zeros(self.n_edges.shape)
            n_edges[index] = self.n_edges[index]
        offsets = self._get_offsets(n_nodes, n_edges)
        kwargs = deepcopy(self.__dict__)
        kwargs['edges'] = edges[:, edge_index]
        kwargs['n_nodes'] = n_nodes
        kwargs['n_edges'] = n_edges
        kwargs['offsets'] = offsets
        kwargs.update(self.data_mask(node_index=node_index, edge_index=edge_index, graph_index=index))
        return self.build(type(self), **kwargs)

    def subbatch(self, index):
        """
        Subgraph of this GraphBatch

        Args:
            index (_type_): index of graphs to be reserved.

        Returns:
            GraphBatch:
        """
        return self.mask_graph(index, compact=True)

    def line_graph(self):
        """
        Construct the line graph of each graph contained in this GraphBatch.
        The node feature of the line graph is inherited from the edge feature of the original graph.

        In the line graph, each node corresponds to an edge in the original graph.
        For a pair of edges (a, b) and (b, c) that share the same intermediate node in the original graph,
        there is a directed edge (a, b) -> (b, c) in the line graph.

        Returns:
            GraphBathch
        """
        node_in, node_out = self.edges
        edge_index = np.arange(self.n_edges)
        edge_in = edge_index[node_out.float().argsort()]
        edge_out = edge_index[node_in.float().argsort()]

        degree_in = node_in.bicount(minlength=self.n_node)
        degree_out = node_out.bincount(minlength=self.n_node)
        size = degree_in * degree_out
        starts = np.repeat(size.cumsum(0) - size, size)
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
        n_nodes = self.n_edges
        n_edges = utils.scatter_add(size, self.node2graph, axis=0, n_axis=self.batch_size)
        offsets = self._get_offsets(n_nodes, n_edges)
        return type(self)(edges=edges, n_nodes=n_nodes, n_edges=n_edges, offsets=offsets,
                          node_type=self.edge_type, node_feat=self.edge_feat)

    def undirected(self, add_inverse=False):
        """
        Flip all the edges to create undirected graphs.

        For knowledge graphs, the flipped edges can either have the original relation or an inverse relation.
        The inverse relation for relation :math:`r` is defined as :math:`|R| + r`.

        Args:
            add_inverse (bool, optional): whether to use inverse relations for flipped edges
        """
        edges = self.edges.copy()
        kwargs = deepcopy(self.__dict__)
        edges = np.flip(edges, 1)
        n_relation = self.n_relation
        if n_relation and add_inverse:
            kwargs['edge_type'] += n_relation
            n_relation = n_relation * 2
        edges = utils.flatten(np.stack([self.edges, edges], axis=-1), 1, 2)
        offsets = self.offsets.expand_dims(-1).repeat(2).flatten()
        index = np.arange(self.n_edges).expand_dims(-1)
        index = np.repeat(index, 2).flatten()

        kwargs = deepcopy(self.__dict__)
        kwargs.update(self.data_mask(edge_index=index))
        kwargs['edges'] = edges
        kwargs['n_edges'] = self.n_edges * 2
        kwargs['n_relation'] = n_relation
        kwargs['offsets'] = offsets
        return self.build(type(self), **kwargs)

    def _get_offsets(self, n_nodes=None, n_edges=None, cum_nodes=None, cum_edges=None):
        """_summary_

        Args:
            n_nodes (array_like, optional): number of nodes for each graph. Defaults to None.
            n_edges (array_like, optional): number of edges for each graph. Defaults to None.
            cum_nodes (array_like, optional): cumsum of n_nodes. Defaults to None.
            cum_edges (array_like, optional): cumsum of n_edges. Defaults to None.

        Returns:
            offset (array_like): offset of edges.
        """
        prepend = 0 if self.detach else ops.zeros(1).int()
        if n_nodes is None:
            n_nodes = cum_nodes.diff(prepend=prepend)
        if n_edges is None:
            n_edges = cum_edges.diff(prepend=prepend)
        if cum_nodes is None:
            cum_nodes = n_nodes.cumsum(0)
        if not self.detach:
            n_edges = n_edges.asnumpy().tolist()
        return (cum_nodes - n_nodes).repeat(n_edges)

    def _masked_n_xs(self, mask, cum_xs):
        """
        Generate new cumsumed data based on the mask

        Args:
            mask (array_like): data mask
            cum_xs (array_like): cumsumed data

        Returns:
            new_cum_xs (array_like): new cumsumed data based on the mask.
        """
        if self.detach:
            x = np.zeros(cum_xs[-1], dtype=int)
        else:
            x = ops.zeros(cum_xs[-1]).int()
        x[mask] = 1
        cum_indexes = x.cumsum(0)
        if self.detach:
            cum_indexes = np.concatenate([np.zeros(1, dtype=int), cum_indexes])
        else:
            cum_indexes = ops.concat([ops.zeros(1, dtype=ms.int32), cum_indexes])
        new_cum_xs = cum_indexes[cum_xs]
        new_num_xs = np.diff(new_cum_xs, prepdend=0) if self.detach else new_cum_xs.diff(prepend=ops.zeros(1).int())
        return new_num_xs


Graph.batch_type = GraphBatch
GraphBatch.item_type = Graph
