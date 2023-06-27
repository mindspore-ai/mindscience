# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
Schnet
"""

from typing import Optional, Callable, Tuple
import math

import mindspore as ms
from mindspore import ops
from mindspore import nn

from .. import core
from .. import util
from ..layers import GaussianSmearing
from ..layers import ShiftedSoftplus
from ..layers import SumAggregation, MeanAggregation


class SchNet(nn.Cell):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (Callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_num_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (ms.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(self,
                 hidden_channels: int = 128,
                 num_filters: int = 128,
                 num_interactions: int = 6,
                 num_gaussians: int = 50,
                 cutoff: float = 10.0,
                 interaction_graph: Optional[Callable] = None,
                 max_neighbors: int = 32,
                 readout: str = 'add',
                 dipole: bool = False,
                 mean: Optional[float] = None,
                 std: Optional[float] = None,
                 atomref: Optional[ms.Tensor] = None):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.aggr = SumAggregation()
        if readout == 'add':
            self.readout = SumAggregation()
        else:
            self.readout = MeanAggregation()
        self.mean = mean
        self.std = std
        self.scale = None

        self.embedding = nn.Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(
                cutoff, max_neighbors)

        self.rbf = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = nn.CellList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.linear = nn.Dense(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()

        self.atomref = None
        if atomref is not None:
            self.atomref = nn.Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')

    def construct(self,
                  coord: ms.Tensor,
                  node_type: ms.Tensor,
                  batch: Optional[ms.Tensor] = None
                  ) -> ms.Tensor:
        r"""
        Args:
            node_type (LongTensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            coord (ms.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = ops.zeros_like(node_type) if batch is None else batch

        node_feat = self.embedding(node_type)
        edge_index, edge_weight = self.interaction_graph(coord, batch)
        edge_feat = self.rbf(edge_weight)

        for interaction in self.interactions:
            node_feat += interaction(node_feat, edge_index, edge_weight, edge_feat)

        node_feat = self.linear(node_feat)
        node_feat = self.act(node_feat)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[node_feat].view(-1, 1)
            m = self.aggr(mass, batch, axis=0)
            c = self.aggr(mass * coord, batch, axis=0) / m
            node_feat = node_feat * (coord - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            node_feat = node_feat * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            node_feat = node_feat + self.atomref(node_feat)
        if batch is not None:
            order = ops.arange(len(batch))
            node2graph = order.repeat(batch.asnumpy().tolist())
            out = self.readout(node2graph, node_feat, axis=0)
        else:
            out = node_feat.sum(axis=0, keepdims=True)

        if self.dipole:
            out = out.norm(axis=-1, keepdims=True)

        if self.scale is not None:
            out = self.scale * out

        return out


class RadiusInteractionGraph(core.Cell):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """

    def __init__(self, cutoff: float = 10.0, max_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

    def construct(self,
                  coord: ms.Tensor,
                  batch: ms.Tensor = None,
                  ) -> Tuple[ms.Tensor, ms.Tensor]:
        r"""
        Args:
            coord (ms.Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(coord, radius=self.cutoff, batch=batch,
                                  max_neighbors=self.max_neighbors)
        row, col = edge_index.T
        edge_weight = (coord[row] - coord[col]).norm(dim=-1)
        return edge_index, edge_weight


class InteractionBlock(core.Cell):
    """_summary_

    Args:
        core (_type_): _description_
    """
    def __init__(self,
                 hidden_channels: int,
                 num_gaussians: int,
                 num_filters: int,
                 cutoff: float):
        super().__init__()
        self.mlp = nn.SequentialCell(
            nn.Dense(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Dense(num_filters, num_filters),
        )
        self.lin1 = nn.Dense(hidden_channels, num_filters)
        self.lin2 = nn.Dense(num_filters, hidden_channels)
        self.conv = GraphConv()
        self.act = ShiftedSoftplus()
        self.lin = nn.Dense(hidden_channels, hidden_channels)
        self.cutoff = cutoff

    def construct(self,
                  node_feat: ms.Tensor,
                  edge_index: ms.Tensor,
                  edge_weight: ms.Tensor,
                  edge_feat: ms.Tensor
                  ) -> ms.Tensor:
        """_summary_

        Args:
            node_feat (ms.Tensor): _description_
            edge_index (ms.Tensor): _description_
            edge_weight (ms.Tensor): _description_
            edge_feat (ms.Tensor): _description_

        Returns:
            ms.Tensor: _description_
        """
        c = 0.5 * (ops.cos(edge_weight * math.pi / self.cutoff) + 1.0)
        w = self.mlp(edge_feat) * c.view(-1, 1)
        node_feat = self.lin1(node_feat)
        node_feat = self.conv(node_feat, edge_index, w)
        node_feat = self.lin2(node_feat)
        node_feat = self.act(node_feat)
        node_feat = self.lin(node_feat)
        return node_feat


class GraphConv(core.Cell):
    """_summary_

    Args:
        core (_type_): _description_
    """
    def message(self, edge_list, node_feat: ms.Tensor, w: ms.Tensor):
        """_summary_

        Args:
            edge_list (_type_): _description_
            node_feat (ms.Tensor): _description_
            W (ms.Tensor): _description_

        Returns:
            _type_: _description_
        """
        node_in = edge_list[:, 0]
        message = node_feat[node_in]
        return message * w

    def aggregate(self, edge_list: ms.Tensor, message: ms.Tensor):
        """_summary_

        Args:
            edge_list (ms.Tensor): _description_
            message (ms.Tensor): _description_

        Returns:
            _type_: _description_
        """
        n_node = util.max(edge_list) + 1
        update = util.scatter_add(message, edge_list[:, 1], axis=0, n_axis=n_node)
        return update

    def update(self, node_feat: ms.Tensor, update: ms.Tensor):
        """update"""
        return node_feat + update

    def construct(self, node_feat, edge_list, edge_feat):
        message = self.message(edge_list, node_feat, edge_feat)
        update = self.aggregate(edge_list, message)
        output = self.update(node_feat, update)
        return output


def radius_graph(coord, radius=3, batch=None, max_neighbors=None):
    """_summary_

    Args:
        coord (_type_): _description_
        radius (int, optional): _description_. Defaults to 3.
        batch (_type_, optional): _description_. Defaults to None.
        max_neighbors (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if batch is None:
        edge_list = knn_graphs(coord, radius, max_neighbors)
    else:
        coords = coord.split(batch.asnumpy().tolist())
        common_map = ops.Map()
        edge_lists = common_map(ops.partial(knn_graphs, radius, max_neighbors), coords)
        n_edges = [len(edge) for edge in edge_lists]
        cum_nodes = batch.cumsum(0)
        offsets = (cum_nodes - batch).repeat(n_edges)
        edge_list = ops.concat(edge_lists, axis=0)
        edge_list += ops.expand_dims(offsets, -1)
    return edge_list


knn_graphs = ops.MultitypeFuncGraph('knn_graph')


@knn_graphs.register('Number', 'Number', 'Tensor')
def knn_graph(radius, max_neighbors, coord):
    """_summary_

    Args:
        radius (_type_): _description_
        max_neighbors (_type_): _description_
        coord (_type_): _description_

    Returns:
        _type_: _description_
    """
    dist = ops.cdist(coord, coord)
    if max_neighbors is None:
        edge_list = ops.nonzero(dist < radius)
    else:
        sorted_dist, neighbors = ops.sort(dist, axis=1)
        mask = sorted_dist < radius
        if max_neighbors < len(coord):
            neighbors = neighbors[:, 1:max_neighbors]
            mask = mask[:, 1:max_neighbors]
        edge_in = ops.nonzero(mask)[:, 0].astype(ms.int32)
        edge_out = neighbors[mask]
        edge_list = ops.stack([edge_in, edge_out], axis=1)
    return edge_list
