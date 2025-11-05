# ============================================================================
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""GNS Molecule."""


from typing import List, Literal, Optional, Dict, Any, Union
from functools import partial

import numpy as np
from mindspore import nn, ops, Tensor, mint
from mindspore.common.initializer import Uniform
import mindspore.ops.operations as P

from mindchemistry.cell.orb.utils import build_mlp

_KEY = "feat"


def mlp_and_layer_norm(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> nn.SequentialCell:
    """Create an MLP followed by layer norm.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dim (int): Hidden dimension.
        n_layers (int): Number of hidden layers.

    Returns:
        nn.SequentialCell: A sequential cell containing the MLP and layer norm.
    """
    layers = build_mlp(
        in_dim,
        [hidden_dim for _ in range(n_layers)],
        out_dim,
    )
    layers.append(nn.LayerNorm((out_dim,)))
    return layers


def get_cutoff(p: int, r: Tensor, r_max: float) -> Tensor:
    """Get the cutoff function for attention.

    Args:
        p (int): Polynomial order.
        r (Tensor): Distance tensor.
        r_max (float): Maximum distance for the cutoff.

    Returns:
        Tensor: Cutoff tensor.
    """
    envelope = 1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * ops.pow(r / r_max, p) + \
        p * (p + 2.0) * ops.pow(r / r_max, p + 1) - \
        (p * (p + 1.0) / 2) * ops.pow(r / r_max, p + 2)
    cutoff = ops.expand_dims(
        ops.where(r < r_max, envelope, ops.zeros_like(envelope)), -1)
    return cutoff


def gaussian_basis_function(
        scalars: Tensor,
        num_bases: Union[Tensor, int],
        radius: Union[Tensor, float],
        scale: Union[Tensor, float] = 1.0,
) -> Tensor:
    """Gaussian basis function applied to a tensor of scalars.

    Args:
        scalars (Tensor): Scalars to compute the gbf on. Shape [num_scalars].
        num_bases (Tensor): The number of bases. An Int.
        radius (Tensor): The largest centre of the bases. A Float.
        scale (Tensor, optional): The width of the gaussians. Defaults to 1.

    Returns:
        Tensor: A tensor of shape [num_scalars, num_bases].
    """
    assert len(scalars.shape) == 1
    gaussian_means = ops.arange(
        0, float(radius), float(radius) / num_bases
    )
    return mint.exp(
        -(scale**2) * (scalars.unsqueeze(1) - gaussian_means.unsqueeze(0)).abs() ** 2
    )


class AtomEmbedding(nn.Cell):
    r"""
    AtomEmbedding Layer.

    This layer initializes atom embeddings based on the atomic number of elements in the periodic table.
    It uses an embedding table initialized with a uniform distribution over the range [-sqrt(3), sqrt(3)].

    Args:
        emb_size (int): Size of the embedding vector for each atom.
        num_elements (int): Number of elements in the periodic table (typically 118 for known elements).

    Inputs:
        - **x** (Tensor) - Input tensor of shape [..., num_atoms], where
          each value represents the atomic number of an atom in the periodic table.

    Outputs:
        - **h** (Tensor) - Output tensor of shape [..., num_atoms, emb_size],
          where each atom's embedding is represented as a vector of size `emb_size`.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, emb_size, num_elements):
        """init
        """
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = nn.Embedding(
            num_elements + 1, emb_size, embedding_table=Uniform(np.sqrt(3)))

    def construct(self, x):
        """construct
        """
        h = self.embeddings(x)
        return h


class Encoder(nn.Cell):
    r"""
    Encoder for Graph Network States (GNS).

    This encoder processes node and edge features using MLPs and layer normalization.
    It concatenates the features of nodes and edges, applies MLPs to update their states,
    and returns the updated features.

    Args:
        num_node_in_features (int): Number of input features for nodes.
        num_node_out_features (int): Number of output features for nodes.
        num_edge_in_features (int): Number of input features for edges.
        num_edge_out_features (int): Number of output features for edges.
        num_mlp_layers (int): Number of MLP layers.
        mlp_hidden_dim (int): Hidden dimension for the MLP.
        node_feature_names (List[str]): List of node feature names.
        edge_feature_names (List[str]): List of edge feature names.

    Inputs:
        - **nodes** (Dict[str, Tensor]) - Dictionary of node features, where keys are feature names
          and values are tensors of shape (num_nodes, num_node_in_features).
        - **edges** (Dict[str, Tensor]) - Dictionary of edge features, where keys are feature names
          and values are tensors of shape (num_edges, num_edge_in_features).

    Outputs:
        - **edges** (Dict[str, Tensor]) - Updated edge features dictionary, where key "feat" contains
          the updated edge features of shape (num_edges, num_edge_out_features).
        - **nodes** (Dict[str, Tensor]) - Updated node features dictionary, where key "feat" contains
          the updated node features of shape (num_nodes, num_node_out_features).

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self,
                 num_node_in_features: int,
                 num_node_out_features: int,
                 num_edge_in_features: int,
                 num_edge_out_features: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int,
                 node_feature_names: List[str],
                 edge_feature_names: List[str]):
        """init
        """
        super().__init__()
        self.node_feature_names = node_feature_names
        self.edge_feature_names = edge_feature_names
        self._node_fn = mlp_and_layer_norm(
            num_node_in_features, num_node_out_features, mlp_hidden_dim, num_mlp_layers)
        self._edge_fn = mlp_and_layer_norm(
            num_edge_in_features, num_edge_out_features, mlp_hidden_dim, num_mlp_layers)

    def construct(self, nodes, edges):
        """construct
        """
        edge_features = ops.cat([edges[k] for k in self.edge_feature_names], axis=-1)
        node_features = ops.cat([nodes[k] for k in self.node_feature_names], axis=-1)

        edges.update({_KEY: self._edge_fn(edge_features)})
        nodes.update({_KEY: self._node_fn(node_features)})
        return edges, nodes


class InteractionNetwork(nn.Cell):
    r"""
    Interaction Network.

    Implements a message passing neural network layer that updates node and edge features based on interactions.
    This layer combines node and edge features, applies MLPs to update their states, and returns the updated features.

    Args:
        num_node_in (int): Number of input features for nodes.
        num_node_out (int): Number of output features for nodes.
        num_edge_in (int): Number of input features for edges.
        num_edge_out (int): Number of output features for edges.
        num_mlp_layers (int): Number of MLP layers.
        mlp_hidden_dim (int): Hidden dimension for the MLP.

    Inputs:
        - **graph_edges** (Dict[str, Tensor]) - Dictionary of edge features, where key "feat" contains
          the edge features of shape (num_edges, num_edge_in).
        - **graph_nodes** (Dict[str, Tensor]) - Dictionary of node features, where key "feat" contains
          the node features of shape (num_nodes, num_node_in).
        - **senders** (Tensor) - Indices of the sender nodes for each edge, shape (num_edges,).
        - **receivers** (Tensor) - Indices of the receiver nodes for each edge, shape (num_edges,).

    Outputs:
        - **edges** (Dict[str, Tensor]) - Updated edge features dictionary, where key "feat" contains
          the updated edge features of shape (num_edges, num_edge_out).
        - **nodes** (Dict[str, Tensor]) - Updated node features dictionary, where key "feat" contains
          the updated node features of shape (num_nodes, num_node_out).

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self,
                 num_node_in: int,
                 num_node_out: int,
                 num_edge_in: int,
                 num_edge_out: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int):
        """init
        """
        super().__init__()
        self._node_mlp = mlp_and_layer_norm(
            num_node_in + num_edge_out, num_node_out, mlp_hidden_dim, num_mlp_layers)
        self._edge_mlp = mlp_and_layer_norm(
            num_node_in + num_node_in + num_edge_in, num_edge_out, mlp_hidden_dim, num_mlp_layers)

    def construct(self, graph_edges, graph_nodes, senders, receivers):
        """construct
        """
        nodes = graph_nodes[_KEY]
        edges = graph_edges[_KEY]

        sent_attributes = ops.gather(nodes, senders, 0)
        received_attributes = ops.gather(nodes, receivers, 0)

        edge_features = ops.cat(
            [edges, sent_attributes, received_attributes], axis=1)
        updated_edges = self._edge_mlp(edge_features)

        received_attributes = ops.scatter_add(
            ops.zeros_like(nodes), receivers, updated_edges)

        node_features = ops.cat([nodes, received_attributes], axis=1)
        updated_nodes = self._node_mlp(node_features)

        nodes = graph_nodes[_KEY] + updated_nodes
        edges = graph_edges[_KEY] + updated_edges

        node_features = {**graph_nodes, _KEY: nodes}
        edge_features = {**graph_edges, _KEY: edges}
        return edge_features, node_features


# pylint: disable=C0301
class AttentionInteractionNetwork(nn.Cell):
    r"""
    Attention interaction network.
    Implements attention-based message passing neural network layer for edge updates in molecular graphs.

    Args:
        num_node_in (int): Number of input node features.
        num_node_out (int): Number of output node features.
        num_edge_in (int): Number of input edge features.
        num_edge_out (int): Number of output edge features.
        num_mlp_layers (int): Number of hidden layers in node and edge update MLPs.
        mlp_hidden_dim (int): Hidden dimension size of MLPs.
        attention_gate (str, optional): Attention gate type, ``"sigmoid"`` or ``"softmax"``. Default: ``"sigmoid"``.
        distance_cutoff (bool, optional): Whether to use distance-based edge cutoff. Default: ``True``.
        polynomial_order (int, optional): Order of polynomial cutoff function. Default: ``4``.
        cutoff_rmax (float, optional): Maximum distance for cutoff. Default: ``6.0``.

    Inputs:
        - **graph_edges** (dict) - Edge feature dictionary, must contain key "feat" with shape :math:`(n_{edges}, num\_edge\_in)`.
        - **graph_nodes** (dict) - Node feature dictionary, must contain key "feat" with shape :math:`(n_{nodes}, num\_node\_in)`.
        - **senders** (Tensor) - Sender node indices for each edge, shape :math:`(n_{edges},)`.
        - **receivers** (Tensor) - Receiver node indices for each edge, shape :math:`(n_{edges},)`.

    Outputs:
        - **edges** (dict) - Updated edge feature dictionary with key "feat" of shape :math:`(n_{edges}, num\_edge\_out)`.
        - **nodes** (dict) - Updated node feature dictionary with key "feat" of shape :math:`(n_{nodes}, num\_node\_out)`.

    Raises:
        ValueError: If `attention_gate` is not "sigmoid" or "softmax".
        ValueError: If edge or node features do not contain the required "feat" key.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindchemistry.cell.orb.gns import AttentionInteractionNetwork
        >>> attn_net = AttentionInteractionNetwork(
        ...     num_node_in=256,
        ...     num_node_out=256,
        ...     num_edge_in=256,
        ...     num_edge_out=256,
        ...     num_mlp_layers=2,
        ...     mlp_hidden_dim=512,
        ... )
        >>> n_atoms = 4
        >>> n_edges = 10
        >>> atomic_numbers = Tensor(np.random.randint(1, 119, size=(n_atoms,), dtype=np.int32))
        >>> atomic_numbers_embedding_np = np.zeros((n_atoms, 118), dtype=np.float32)
        >>> for i, num in enumerate(atomic_numbers.asnumpy()):
        ...     atomic_numbers_embedding_np[i, num - 1] = 1.0
        >>> node_features = {
        ...     "atomic_numbers": atomic_numbers,
        ...     "atomic_numbers_embedding": Tensor(atomic_numbers_embedding_np),
        ...     "positions": Tensor(np.random.randn(n_atoms, 3).astype(np.float32)),
        ...     "feat": Tensor(np.random.randn(n_atoms, 256).astype(np.float32))
        ... }
        >>> edge_features = {
        ...     "vectors": Tensor(np.random.randn(n_edges, 3).astype(np.float32)),
        ...     "r": Tensor(np.abs(np.random.randn(n_edges).astype(np.float32) * 10)),
        ...     "feat": Tensor(np.random.randn(n_edges, 256).astype(np.float32))
        ... }
        >>> senders = Tensor(np.random.randint(0, n_atoms, size=(n_edges,), dtype=np.int32))
        >>> receivers = Tensor(np.random.randint(0, n_atoms, size=(n_edges,), dtype=np.int32))
        >>> edges, nodes = attn_net(
        ...     edge_features,
        ...     node_features,
        ...     senders,
        ...     receivers,
        ... )
        >>> print(edges["feat"].shape, nodes["feat"].shape)
        (10, 256) (4, 256)
    """

    def __init__(self,
                 num_node_in: int,
                 num_node_out: int,
                 num_edge_in: int,
                 num_edge_out: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int,
                 attention_gate: Literal["sigmoid", "softmax"] = "sigmoid",
                 distance_cutoff: bool = True,
                 polynomial_order: Optional[int] = 4,
                 cutoff_rmax: Optional[float] = 6.0):
        """init
        """
        super().__init__()
        self._num_node_in = num_node_in
        self._num_node_out = num_node_out
        self._num_edge_in = num_edge_in
        self._num_edge_out = num_edge_out
        self._num_mlp_layers = num_mlp_layers
        self._mlp_hidden_dim = mlp_hidden_dim
        self._node_mlp = mlp_and_layer_norm(
            num_node_in + num_edge_out + num_edge_out, num_node_out, mlp_hidden_dim, num_mlp_layers)
        self._edge_mlp = mlp_and_layer_norm(
            num_node_in + num_node_in + num_edge_in, num_edge_out, mlp_hidden_dim, num_mlp_layers)
        self._receive_attn = nn.Dense(num_edge_in, 1)
        self._send_attn = nn.Dense(num_edge_in, 1)
        self._distance_cutoff = distance_cutoff
        self._r_max = cutoff_rmax
        self._polynomial_order = polynomial_order
        self._attention_gate = attention_gate

        self.scatter_add = P.TensorScatterAdd()

    def construct(self, graph_edges, graph_nodes, senders, receivers):
        """construct
        """
        nodes = graph_nodes[_KEY]
        edges = graph_edges[_KEY]

        p = self._polynomial_order
        r_max = self._r_max
        r = graph_edges['r']
        cutoff = get_cutoff(p, r, r_max)

        sent_attributes = ops.gather(nodes, senders, 0)
        received_attributes = ops.gather(nodes, receivers, 0)

        if self._attention_gate == "softmax":
            receive_attn = ops.softmax(self._receive_attn(edges), axis=0)
            send_attn = ops.softmax(self._send_attn(edges), axis=0)
        else:
            receive_attn = ops.sigmoid(self._receive_attn(edges))
            send_attn = ops.sigmoid(self._send_attn(edges))

        if self._distance_cutoff:
            receive_attn = receive_attn * cutoff
            send_attn = send_attn * cutoff

        edge_features = ops.cat(
            [edges, sent_attributes, received_attributes], axis=1)
        updated_edges = self._edge_mlp(edge_features)

        if senders.ndim < 2:
            senders = senders.unsqueeze(-1)
        sent_attributes = self.scatter_add(
            ops.zeros_like(nodes), senders, updated_edges * send_attn)
        if receivers.ndim < 2:
            receivers = receivers.unsqueeze(-1)
        received_attributes = self.scatter_add(
            ops.zeros_like(nodes), receivers, updated_edges * receive_attn)

        node_features = ops.cat(
            [nodes, received_attributes, sent_attributes], axis=1)
        updated_nodes = self._node_mlp(node_features)

        nodes = graph_nodes[_KEY] + updated_nodes
        edges = graph_edges[_KEY] + updated_edges

        node_features = {**graph_nodes, _KEY: nodes}
        edge_features = {**graph_edges, _KEY: edges}
        return edge_features, node_features

class Decoder(nn.Cell):
    r"""
    Decoder for Graph Network States (GNS).

    This decoder processes node features using an MLP to produce predictions.
    It takes the node features as input and outputs updated node features with predictions.

    Args:
        num_node_in (int): Number of input features for nodes.
        num_node_out (int): Number of output features for nodes.
        num_mlp_layers (int): Number of MLP layers.
        mlp_hidden_dim (int): Hidden dimension for the MLP.
        batch_norm (bool, optional): Whether to apply batch normalization. Defaults to False.

    Inputs:
        - **graph_nodes** (Dict[str, Tensor]) - Dictionary of node features, where key "feat" contains
          the node features of shape (num_nodes, num_node_in).

    Outputs:
        - **graph_nodes** (Dict[str, Tensor]) - Updated node features dictionary, where key "pred" contains
          the predicted node features of shape (num_nodes, num_node_out).

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self,
                 num_node_in: int,
                 num_node_out: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int,
                 batch_norm: bool = False):
        """Initialization.
        Args:
            num_node_in (int): Number of input features for nodes.
            num_node_out (int): Number of output features for nodes.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): Hidden dimension for the MLP.
            batch_norm (bool, optional): Whether to apply batch normalization. Defaults to False.
        """
        super().__init__()
        seq = build_mlp(
            num_node_in,
            [mlp_hidden_dim for _ in range(num_mlp_layers)],
            num_node_out,
        )
        if batch_norm:
            seq.append(nn.BatchNorm1d(num_node_out))
        self.node_fn = nn.SequentialCell(seq)

    def construct(self, graph_nodes):
        """Forward pass of the decoder.
        Args:
            graph_nodes (Dict[str, Tensor]): Dictionary of node features.
        Returns:
            Dict[str, Tensor]: Updated node features with predictions.
        """
        nodes = graph_nodes[_KEY]
        updated = self.node_fn(nodes)
        return {**graph_nodes, "pred": updated}


# pylint: disable=C0301
class MoleculeGNS(nn.Cell):
    r"""
    Molecular graph neural network.
    Implements flexible modular graph neural network for molecular property prediction based on message passing
    with attention or other interaction mechanisms. Supports node and edge embeddings, multiple message passing
    steps, and customizable interaction layers for complex molecular graphs.

    Args:
        num_node_in_features (int): Number of input features per node.
        num_node_out_features (int): Number of output features per node.
        num_edge_in_features (int): Number of input features per edge.
        latent_dim (int): Latent dimension for node and edge representations.
        num_message_passing_steps (int): Number of message passing layers.
        num_mlp_layers (int): Number of hidden layers in node and edge update MLPs.
        mlp_hidden_dim (int): Hidden dimension size of MLPs.
        node_feature_names (List[str]): List of node feature keys to use from input dictionary.
        edge_feature_names (List[str]): List of edge feature keys to use from input dictionary.
        use_embedding (bool, optional): Whether to use atomic number embedding for nodes. Default: ``True``.
        interactions (str, optional): Type of interaction layer to use (e.g., ``"simple_attention"``). Default: ``"simple_attention"``.
        interaction_params (Optional[Dict[str, Any]], optional): Parameters for interaction layer, e.g., cutoff,
            polynomial order, gate type. Default: ``None``.

    Inputs:
        - **edge_features** (dict) - Edge feature dictionary, must contain keys specified in `edge_feature_names`.
        - **node_features** (dict) - Node feature dictionary, must contain keys specified in `node_feature_names`.
        - **senders** (Tensor) - Sender node indices for each edge, shape :math:`(n_{edges},)`.
        - **receivers** (Tensor) - Receiver node indices for each edge, shape :math:`(n_{edges},)`.

    Outputs:
        - **edges** (dict) - Updated edge feature dictionary with key "feat" of shape :math:`(n_{edges}, latent\_dim)`.
        - **nodes** (dict) - Updated node feature dictionary with key "feat" of shape :math:`(n_{nodes}, latent\_dim)`.

    Raises:
        ValueError: If required feature keys are missing in `edge_features` or `node_features`.
        ValueError: If `interactions` is not a supported type.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindchemistry.cell.orb.gns import MoleculeGNS
        >>> gns_model = MoleculeGNS(
        ...     num_node_in_features=256,
        ...     num_node_out_features=3,
        ...     num_edge_in_features=23,
        ...     latent_dim=256,
        ...     interactions="simple_attention",
        ...     interaction_params={
        ...         "distance_cutoff": True,
        ...         "polynomial_order": 4,
        ...         "cutoff_rmax": 6,
        ...         "attention_gate": "sigmoid",
        ...     },
        ...     num_message_passing_steps=15,
        ...     num_mlp_layers=2,
        ...     mlp_hidden_dim=512,
        ...     use_embedding=True,
        ...     node_feature_names=["feat"],
        ...     edge_feature_names=["feat"],
        ... )
        >>> n_atoms = 4
        >>> n_edges = 10
        >>> atomic_numbers = Tensor(np.random.randint(1, 119, size=(n_atoms,), dtype=np.int32))
        >>> atomic_numbers_embedding_np = np.zeros((n_atoms, 118), dtype=np.float32)
        >>> for i, num in enumerate(atomic_numbers.asnumpy()):
        ...     atomic_numbers_embedding_np[i, num - 1] = 1.0
        >>> node_features = {
        ...     "atomic_numbers": atomic_numbers,
        ...     "atomic_numbers_embedding": Tensor(atomic_numbers_embedding_np),
        ...     "positions": Tensor(np.random.randn(n_atoms, 3).astype(np.float32)),
        ...     "feat": Tensor(np.random.randn(n_atoms, 256).astype(np.float32))
        ... }
        >>> edge_features = {
        ...     "vectors": Tensor(np.random.randn(n_edges, 3).astype(np.float32)),
        ...     "r": Tensor(np.abs(np.random.randn(n_edges).astype(np.float32) * 10)),
        ...     "feat": Tensor(np.random.randn(n_edges, 256).astype(np.float32))
        ... }
        >>> senders = Tensor(np.random.randint(0, n_atoms, size=(n_edges,), dtype=np.int32))
        >>> receivers = Tensor(np.random.randint(0, n_atoms, size=(n_edges,), dtype=np.int32))
        >>> edges, nodes = gns_model(
        ...     edge_features,
        ...     node_features,
        ...     senders,
        ...     receivers,
        ... )
        >>> print(edges["feat"].shape, nodes["feat"].shape)
        (10, 256) (4, 256)
    """

    def __init__(self,
                 num_node_in_features: int,
                 num_node_out_features: int,
                 num_edge_in_features: int,
                 latent_dim: int,
                 num_message_passing_steps: int,
                 num_mlp_layers: int,
                 mlp_hidden_dim: int,
                 node_feature_names: List[str],
                 edge_feature_names: List[str],
                 use_embedding: bool = True,
                 interactions: Literal["default",
                                       "simple_attention"] = "simple_attention",
                 interaction_params: Optional[Dict[str, Any]] = None):
        """init
        """
        super().__init__()
        self._encoder = Encoder(
            num_node_in_features=num_node_in_features,
            num_node_out_features=latent_dim,
            num_edge_in_features=num_edge_in_features,
            num_edge_out_features=latent_dim,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            node_feature_names=node_feature_names,
            edge_feature_names=edge_feature_names
        )
        if interactions == "default":
            InteractionNetworkClass = InteractionNetwork
        elif interactions == "simple_attention":
            InteractionNetworkClass = AttentionInteractionNetwork
        self.num_message_passing_steps = num_message_passing_steps
        if interaction_params is None:
            interaction_params = {}
        self.gnn_stacks = nn.CellList([
            InteractionNetworkClass(
                num_node_in=latent_dim,
                num_node_out=latent_dim,
                num_edge_in=latent_dim,
                num_edge_out=latent_dim,
                num_mlp_layers=num_mlp_layers,
                mlp_hidden_dim=mlp_hidden_dim,
                **interaction_params
            ) for _ in range(self.num_message_passing_steps)
        ])
        self._decoder = Decoder(
            num_node_in=latent_dim,
            num_node_out=num_node_out_features,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim
        )
        self.rbf = partial(gaussian_basis_function, num_bases=20, radius=10.0)
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.atom_emb = AtomEmbedding(latent_dim, 118)

    def construct(self, edge_features, node_features, senders, receivers):
        """construct
        """
        edge_features = self.featurize_edges(edge_features)
        node_features = self.featurize_nodes(node_features)
        edges, nodes = self._encoder(node_features, edge_features)
        for gnn in self.gnn_stacks:
            edges, nodes = gnn(edges, nodes, senders, receivers)
        nodes = self._decoder(nodes)
        return edges, nodes

    def featurize_nodes(self, node_features):
        """Featurize the nodes of a graph.

        Args:
            node_features (Dict[str, Tensor]): Dictionary of node features.

        Returns:
            Dict[str, Tensor]: Updated node features with atomic embeddings.
        """
        one_hot_atomic = ops.OneHot()(
            node_features["atomic_numbers"], 118, Tensor(1.0), Tensor(0.0)
        )
        if self.use_embedding:
            atomic_embedding = self.atom_emb(node_features["atomic_numbers"])
        else:
            atomic_embedding = one_hot_atomic

        node_features = {**node_features, **{_KEY: atomic_embedding}}
        return node_features

    def featurize_edges(self, edge_features):
        """Featurize the edges of a graph.

        Args:
            edge_features (Dict[str, Tensor]): Dictionary of edge features.

        Returns:
            Dict[str, Tensor]: Updated edge features with radial basis functions and unit vectors.
        """
        lengths = ops.norm(edge_features['vectors'], dim=1)
        non_zero_divisor = ops.where(
            lengths == 0, ops.ones_like(lengths), lengths)
        unit_vectors = edge_features['vectors'] / ops.expand_dims(non_zero_divisor, 1)
        rbfs = self.rbf(lengths)
        edges = ops.cat([rbfs, unit_vectors], axis=1)

        edge_features = {**edge_features, **{_KEY: edges}}
        return edge_features
