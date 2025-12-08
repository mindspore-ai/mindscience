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
"""Orb GraphRegressor."""

from typing import Literal, Optional, Union
import numpy

import mindspore as ms
from mindspore import Parameter, ops, Tensor, mint

from models.gns import _KEY, MoleculeGNS
from models.utils import (
    aggregate_nodes,
    build_mlp,
    REFERENCE_ENERGIES,
)


class LinearReferenceEnergy(ms.nn.Cell):
    r"""
    Linear reference energy (no bias term).

    This class implements a linear reference energy model that can be used
    to compute the reference energy for a given set of atomic numbers.

    Args:
        weight_init (numpy.ndarray, optional): Initial weights for the linear layer.
            If not provided, the weights will be initialized randomly.
        trainable (bool, optional): Whether the weights are trainable or not.
            If not provided, the weights will be trainable by default.

    Inputs:
        - **atom_types** (Tensor) - A tensor of atomic numbers of shape (n_atoms,).
        - **n_node** (Tensor) - A tensor of shape (n_graphs,) containing the number of nodes in each graph.

    Outputs:
        - **Tensor** - A tensor of shape (n_graphs, 1) containing the reference energy.

    Raises:
        ValueError: If the input tensor shapes are not compatible with the expected shapes.
        TypeError: If the input types are not compatible with the expected types.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(
            self,
            weight_init: Optional[numpy.ndarray] = None,
            trainable: Optional[bool] = None,
    ):
        """init
        """
        super().__init__()

        if trainable is None:
            trainable = weight_init is None

        self.linear = ms.nn.Dense(118, 1, has_bias=False)
        if weight_init is not None:
            self.linear.weight.set_data(Tensor(weight_init, dtype=ms.float32).reshape(1, 118))
        if not trainable:
            self.linear.weight.requires_grad = False

    def construct(self, atom_types: Tensor, n_node: Tensor):
        """construct
        """
        one_hot_atomic = ops.OneHot()(atom_types, 118, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))

        reduced = aggregate_nodes(one_hot_atomic, n_node, reduction="sum")
        return self.linear(reduced)


class ScalarNormalizer(ms.nn.Cell):
    r"""
    Scalar normalizer that learns mean and std from data.

    NOTE: Multi-dimensional tensors are flattened before updating
    the running mean/std. This is desired behaviour for force targets.

    Args:
        init_mean (Tensor or float, optional): Initial mean value for normalization.
            If not provided, defaults to 0.0.
        init_std (Tensor or float, optional): Initial standard deviation value for normalization.
            If not provided, defaults to 1.0.
        init_num_batches (int, optional): Initial number of batches for normalization.
            If not provided, defaults to 1000.

    Inputs:
        - **x** (Tensor) - A tensor of shape (n_samples, n_features) to normalize.

    Outputs:
        - **Tensor** - A tensor of the same shape as x, normalized by the running mean and std.

    Raises:
        ValueError: If the input tensor is not of the expected shape.
        TypeError: If the input types are not compatible with the expected types.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(
            self,
            init_mean: Optional[Union[Tensor, float]] = None,
            init_std: Optional[Union[Tensor, float]] = None,
            init_num_batches: Optional[int] = 1000,
    ):
        """init
        """
        super().__init__()
        self.bn = mint.nn.BatchNorm1d(1, affine=False, momentum=None)
        self.bn.running_mean = Parameter(Tensor([0], ms.float32))
        self.bn.running_var = Parameter(Tensor([1], ms.float32))
        self.bn.num_batches_tracked = Parameter(Tensor([1000], ms.float32), requires_grad=False)
        self.stastics = {
            "running_mean": init_mean if init_mean is not None else 0.0,
            "running_var": init_std**2 if init_std is not None else 1.0,
            "num_batches_tracked": init_num_batches if init_num_batches is not None else 1000,
        }

    def construct(self, x: Tensor):
        """construct
        """
        if self.training:
            self.bn(x.view(-1, 1))
        if hasattr(self, "running_mean"):
            return (x - self.running_mean) / mint.sqrt(self.running_var)
        return (x - self.bn.running_mean) / mint.sqrt(self.bn.running_var)

    def inverse(self, x: Tensor):
        """Reverse the construct normalization.

        Args:
            x: A tensor of shape (n_samples, n_features) to inverse normalize.

        Returns:
            A tensor of the same shape as x, inverse normalized by the running mean and std.
        """
        if hasattr(self, "running_mean"):
            return x * mint.sqrt(self.running_var) + self.running_mean
        return x * mint.sqrt(self.bn.running_var) + self.bn.running_mean

# pylint: disable=C0301
class NodeHead(ms.nn.Cell):
    r"""
    Node-level prediction head.

    Implements neural network head for predicting node-level properties from node features. This head can be
    added to base models to enable auxiliary tasks during pretraining or added in fine-tuning steps.

    Args:
        latent_dim (int): Input feature dimension for each node.
        num_mlp_layers (int): Number of hidden layers in MLP.
        mlp_hidden_dim (int): Hidden dimension size of MLP.
        target_property_dim (int): Output dimension of node-level target property.
        dropout (Optional[float], optional): Dropout rate for MLP. Default: ``None``.
        remove_mean (bool, optional): If True, remove mean from output, typically used for force prediction.
            Default: ``True``.

    Inputs:
        - **node_features** (dict) - Node feature dictionary, must contain key "feat" with shape :math:`(n_{nodes}, latent\_dim)`.
        - **n_node** (Tensor) - Number of nodes in graph, shape :math:`(1,)`.

    Outputs:
        - **output** (dict) - Dictionary containing key "node_pred" with value of shape :math:`(n_{nodes}, target\_property\_dim)`.

    Raises:
        ValueError: If required feature keys are missing in `node_features`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindchemistry.cell.orb.gns import NodeHead
        >>> node_head = NodeHead(
        ...     latent_dim=256,
        ...     num_mlp_layers=1,
        ...     mlp_hidden_dim=256,
        ...     target_property_dim=3,
        ...     remove_mean=True,
        ... )
        >>> n_atoms = 4
        >>> n_node = Tensor([n_atoms], mindspore.int32)
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
        >>> output = node_head(node_features, n_node)
        >>> print(output['node_pred'].shape)
        (4, 3)
    """
    def __init__(
            self,
            latent_dim: int,
            num_mlp_layers: int,
            mlp_hidden_dim: int,
            target_property_dim: int,
            dropout: Optional[float] = None,
            remove_mean: bool = True,
    ):
        """init
        """
        super().__init__()
        self.target_property_dim = target_property_dim
        self.normalizer = ScalarNormalizer()

        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target_property_dim,
            dropout=dropout,
        )

        self.remove_mean = remove_mean

    def construct(self, node_features, n_node):
        """construct
        """
        feat = node_features[_KEY]
        pred = self.mlp(feat)
        if self.remove_mean:
            system_means = aggregate_nodes(
                pred, n_node, reduction="mean"
            )
            node_broadcasted_means = mint.repeat_interleave(
                system_means, n_node, dim=0
            )
            pred = pred - node_broadcasted_means
        res = {"node_pred": pred}
        return res

    def predict(self, node_features, n_node):
        """Predict node-level attributes.

        Args:
            node_features: Node features tensor of shape (n_nodes, latent_dim).
            n_node: Number of nodes in the graph.

        Returns:
            node_pred: Node-level predictions of shape (n_nodes, target_property_dim).
        """
        out = self(node_features, n_node)
        pred = out["node_pred"]
        return self.normalizer.inverse(pred)


# pylint: disable=C0301
class GraphHead(ms.nn.Cell):
    r"""
    Graph-level prediction head. Implements graph-level prediction head that can be attached to base models
    for predicting graph-level properties (e.g., stress tensor) from node features using aggregation and MLP.

    Args:
        latent_dim (int): Input feature dimension for each node.
        num_mlp_layers (int): Number of hidden layers in MLP.
        mlp_hidden_dim (int): Hidden dimension size of MLP.
        target_property_dim (int): Output dimension of graph-level property.
        node_aggregation (str, optional): Aggregation method for node predictions, e.g., ``"mean"`` or ``"sum"``. Default: ``"mean"``.
        dropout (Optional[float], optional): Dropout rate for MLP. Default: ``None``.
        compute_stress (bool, optional): Whether to compute and output stress tensor. Default: ``False``.

    Inputs:
        - **node_features** (dict) - Node feature dictionary, must contain key "feat" with shape :math:`(n_{nodes}, latent\_dim)`.
        - **n_node** (Tensor) - Number of nodes in graph, shape :math:`(1,)`.

    Outputs:
        - **output** (dict) - Dictionary containing key "stress_pred" with value of shape :math:`(1, target\_property\_dim)`.

    Raises:
        ValueError: If required feature keys are missing in `node_features`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindchemistry.cell.orb.gns import GraphHead
        >>> graph_head = GraphHead(
        ...     latent_dim=256,
        ...     num_mlp_layers=1,
        ...     mlp_hidden_dim=256,
        ...     target_property_dim=6,
        ...     compute_stress=True,
        ... )
        >>> n_atoms = 4
        >>> n_node = Tensor([n_atoms], mindspore.int32)
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
        >>> output = graph_head(node_features, n_node)
        >>> print(output['stress_pred'].shape)
        (1, 6)
    """

    def __init__(
            self,
            latent_dim: int,
            num_mlp_layers: int,
            mlp_hidden_dim: int,
            target_property_dim: int,
            node_aggregation: Literal["sum", "mean"] = "mean",
            dropout: Optional[float] = None,
            compute_stress: Optional[bool] = False,
    ):
        """init
        """
        super().__init__()
        self.target_property_dim = target_property_dim
        self.normalizer = ScalarNormalizer()

        self.node_aggregation = node_aggregation
        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target_property_dim,
            dropout=dropout,
        )
        self.output_activation = ops.Identity()
        self.compute_stress = compute_stress

    def construct(self, node_features, n_node):
        """construct
        """
        feat = node_features[_KEY]

        # aggregate to get a tensor of shape (num_graphs, latent_dim)
        mlp_input = aggregate_nodes(
            feat,
            n_node,
            reduction=self.node_aggregation,
        )

        pred = self.mlp(mlp_input)
        if self.compute_stress:
            # name the stress prediction differently
            res = {"stress_pred": pred}
        else:
            res = {"graph_pred": pred}
        return res

    def predict(self, node_features, n_node, atomic_numbers=None):
        """Predict graph-level attributes.

        Args:
            node_features: Node features tensor
            n_node: Number of nodes
            atomic_numbers: Optional atomic numbers for reference energy calculation

        Returns:
            probs: Graph-level predictions of shape (n_graphs, target_property_dim).
            If compute_stress is True, this will be the stress tensor.
            If compute_stress is False, this will be the graph-level property (e.g., energy).
        """
        pred = self(node_features, n_node)
        if self.compute_stress:
            logits = pred["stress_pred"].squeeze(-1)
        else:
            assert atomic_numbers is not None, "atomic_numbers must be provided for graph prediction"
            logits = pred["graph_pred"].squeeze(-1)
        probs = self.output_activation(logits)
        probs = self.normalizer.inverse(probs)
        return probs


# pylint: disable=C0301
class EnergyHead(GraphHead):
    r"""
    Graph-level energy prediction head.
    Implements neural network head for predicting total energy or per-atom average energy of molecular graphs.
    Supports node-level aggregation, reference energy offset, and flexible output modes.

    Args:
        latent_dim (int): Input feature dimension for each node.
        num_mlp_layers (int): Number of hidden layers in MLP.
        mlp_hidden_dim (int): Hidden dimension size of MLP.
        target_property_dim (int): Output dimension of energy property (typically 1).
        predict_atom_avg (bool, optional): Whether to predict per-atom average energy instead of total energy. Default: ``True``.
        reference_energy_name (str, optional): Reference energy name for offset, e.g., ``"vasp-shifted"``. Default: ``"mp-traj-d3"``.
        train_reference (bool, optional): Whether to train reference energy as learnable parameter. Default: ``False``.
        dropout (Optional[float], optional): Dropout rate for MLP. Default: ``None``.
        node_aggregation (str, optional): Aggregation method for node predictions, e.g., ``"mean"`` or ``"sum"``. Default: ``None``.

    Inputs:
        - **node_features** (dict) - Node feature dictionary, must contain key "feat" with shape :math:`(n_{nodes}, latent\_dim)`.
        - **n_node** (Tensor) - Number of nodes in graph, shape :math:`(1,)`.

    Outputs:
        - **output** (dict) - Dictionary containing key "graph_pred" with value of shape :math:`(1, target\_property\_dim)`.

    Raises:
        ValueError: If required feature keys are missing in `node_features`.
        ValueError: If `node_aggregation` is not a supported type.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindchemistry.cell.orb.gns import EnergyHead
        >>> energy_head = EnergyHead(
        ...     latent_dim=256,
        ...     num_mlp_layers=1,
        ...     mlp_hidden_dim=256,
        ...     target_property_dim=1,
        ...     node_aggregation="mean",
        ...     reference_energy_name="vasp-shifted",
        ...     train_reference=True,
        ...     predict_atom_avg=True,
        ... )
        >>> n_atoms = 4
        >>> n_node = Tensor([n_atoms], mindspore.int32)
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
        >>> output = energy_head(node_features, n_node)
        >>> print(output['graph_pred'].shape)
        (1, 1)
    """

    def __init__(
            self,
            latent_dim: int,
            num_mlp_layers: int,
            mlp_hidden_dim: int,
            target_property_dim: int,
            predict_atom_avg: bool = True,
            reference_energy_name: str = "mp-traj-d3",
            train_reference: bool = False,
            dropout: Optional[float] = None,
            node_aggregation: Optional[str] = "mean",
    ):
        """init
        """
        ref = REFERENCE_ENERGIES[reference_energy_name]

        super().__init__(
            latent_dim=latent_dim,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            target_property_dim=target_property_dim,
            node_aggregation=node_aggregation,
            dropout=dropout,
        )
        self.reference = LinearReferenceEnergy(
            weight_init=ref.coefficients, trainable=train_reference
        )
        self.atom_avg = predict_atom_avg

    def predict(self, node_features, n_node, atomic_numbers=None):
        """Predict energy.

        Args:
            node_features: Node features tensor
            n_node: Number of nodes
            atomic_numbers: Optional atomic numbers for reference energy calculation

        Returns:
            graph_pred: Energy prediction
        """
        if atomic_numbers is None:
            raise ValueError("atomic_numbers is required for energy prediction")

        pred = self(node_features, n_node)["graph_pred"]
        pred = self.normalizer.inverse(pred).squeeze(-1)
        if self.atom_avg:
            pred = pred * n_node
        pred = pred + self.reference(atomic_numbers, n_node)
        return pred


# pylint: disable=C0301
class Orb(ms.nn.Cell):
    r"""
    Orb graph regressor.
    Combines a pretrained base model (e.g., MoleculeGNS) with optional node, graph, and stress regression heads, supporting
    fine-tuning or feature extraction workflows.

    Args:
        model (MoleculeGNS): Pretrained or randomly initialized base model for message passing and feature extraction.
        node_head (NodeHead, optional): Regression head for node-level property prediction. Default: ``None``.
        graph_head (GraphHead, optional): Regression head for graph-level property prediction (e.g., energy). Default: ``None``.
        stress_head (GraphHead, optional): Regression head for stress prediction. Default: ``None``.
        model_requires_grad (bool, optional): Whether to fine-tune the base model (True) or freeze its parameters (False). Default: ``True``.
        cutoff_layers (int, optional): If provided, only use the first ``cutoff_layers`` message passing layers of the base model.
            Default: ``None``.

    Inputs:
        - **edge_features** (dict) - Edge feature dictionary (e.g., `{"vectors": Tensor, "r": Tensor}`).
        - **node_features** (dict) - Node feature dictionary (e.g., `{"atomic_numbers": Tensor, ...}`).
        - **senders** (Tensor) - Sender node indices for each edge. Shape: :math:`(n_{edges},)`.
        - **receivers** (Tensor) - Receiver node indices for each edge. Shape: :math:`(n_{edges},)`.
        - **n_node** (Tensor) - Number of nodes for each graph in the batch. Shape: :math:`(n_{graphs},)`.

    Outputs:
        - **output** (dict) - Dictionary containing:
          - **edges** (dict) - Edge features after message passing, e.g., `{..., "feat": Tensor}`.
          - **nodes** (dict) - Node features after message passing, e.g., `{..., "feat": Tensor}`.
          - **graph_pred** (Tensor) - Graph-level predictions, e.g., energy. Shape: :math:`(n_{graphs}, target\_property\_dim)`.
          - **node_pred** (Tensor) - Node-level predictions. Shape: :math:`(n_{nodes}, target\_property\_dim)`.
          - **stress_pred** (Tensor) - Stress predictions (if stress_head is provided). Shape: :math:`(n_{graphs}, 6)`.

    Raises:
        ValueError: If neither node_head nor graph_head is provided.
        ValueError: If cutoff_layers exceeds the number of message passing steps in the base model.
        ValueError: If atomic_numbers is not provided when graph_head is required.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindchemistry.cell.orb import Orb, MoleculeGNS, EnergyHead, NodeHead, GraphHead
        >>> Orb = Orb(
        ...     model=MoleculeGNS(
        ...         num_node_in_features=256,
        ...         num_node_out_features=3,
        ...         num_edge_in_features=23,
        ...         latent_dim=256,
        ...         interactions="simple_attention",
        ...         interaction_params={
        ...             "distance_cutoff": True,
        ...             "polynomial_order": 4,
        ...             "cutoff_rmax": 6,
        ...             "attention_gate": "sigmoid",
        ...         },
        ...         num_message_passing_steps=15,
        ...         num_mlp_layers=2,
        ...         mlp_hidden_dim=512,
        ...         use_embedding=True,
        ...         node_feature_names=["feat"],
        ...         edge_feature_names=["feat"],
        ...     ),
        ...     graph_head=EnergyHead(
        ...         latent_dim=256,
        ...         num_mlp_layers=1,
        ...         mlp_hidden_dim=256,
        ...         target_property_dim=1,
        ...         node_aggregation="mean",
        ...         reference_energy_name="vasp-shifted",
        ...         train_reference=True,
        ...         predict_atom_avg=True,
        ...     ),
        ...     node_head=NodeHead(
        ...         latent_dim=256,
        ...         num_mlp_layers=1,
        ...         mlp_hidden_dim=256,
        ...         target_property_dim=3,
        ...         remove_mean=True,
        ...     ),
        ...     stress_head=GraphHead(
        ...         latent_dim=256,
        ...         num_mlp_layers=1,
        ...         mlp_hidden_dim=256,
        ...         target_property_dim=6,
        ...         compute_stress=True,
        ...     ),
        ... )
        >>> n_atoms = 4
        >>> n_edges = 10
        >>> n_node = Tensor([n_atoms], mindspore.int32)
        >>> atomic_numbers = Tensor(np.random.randint(1, 119, size=(n_atoms,), dtype=np.int32))
        >>> atomic_numbers_embedding_np = np.zeros((n_atoms, 118), dtype=np.float32)
        >>> for i, num in enumerate(atomic_numbers.asnumpy()):
        ...     atomic_numbers_embedding_np[i, num - 1] = 1.0
        >>> node_features = {
        ...     "atomic_numbers": atomic_numbers,
        ...     "atomic_numbers_embedding": Tensor(atomic_numbers_embedding_np),
        ...     "positions": Tensor(np.random.randn(n_atoms, 3).astype(np.float32))
        ... }
        >>> edge_features = {
        ...     "vectors": Tensor(np.random.randn(n_edges, 3).astype(np.float32)),
        ...     "r": Tensor(np.abs(np.random.randn(n_edges).astype(np.float32) * 10))
        ... }
        >>> senders = Tensor(np.random.randint(0, n_atoms, size=(n_edges,), dtype=np.int32))
        >>> receivers = Tensor(np.random.randint(0, n_atoms, size=(n_edges,), dtype=np.int32))
        >>> output = Orb(edge_features, node_features, senders, receivers, n_node)
        >>> print(output['graph_pred'].shape, output['node_pred'].shape, output['stress_pred'].shape)
        (1, 1) (4, 3) (1, 6)
    """

    def __init__(
            self,
            model: MoleculeGNS,
            node_head: Optional[NodeHead] = None,
            graph_head: Optional[GraphHead] = None,
            stress_head: Optional[GraphHead] = None,
            model_requires_grad: bool = True,
            cutoff_layers: Optional[int] = None,
    ):
        """init
        """
        super().__init__()

        if (node_head is None) and (graph_head is None):
            raise ValueError("Must provide at least one node/graph head.")
        self.node_head = node_head
        self.graph_head = graph_head
        self.stress_head = stress_head
        self.cutoff_layers = cutoff_layers

        self.model = model

        if self.cutoff_layers is not None:
            if self.cutoff_layers > self.model.num_message_passing_steps:
                raise ValueError(
                    f"cutoff_layers ({self.cutoff_layers}) must be less than or equal to"
                    f" the number of message passing steps ({self.model.num_message_passing_steps})"
                )
            self.model.gnn_stacks = self.model.gnn_stacks[: self.cutoff_layers]
            self.model.num_message_passing_steps = self.cutoff_layers

        self.model_requires_grad = model_requires_grad

        if not model_requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False


    def predict(self, edge_features, node_features, senders, receivers, n_node, atomic_numbers):
        """Predict node and/or graph level attributes.

        Args:
            edge_features: A dictionary, e.g., `{"vectors": Tensor, "r": Tensor}`.
            node_features: A dictionary, e.g., `{"atomic_numbers": Tensor, "positions": Tensor,
                "atomic_numbers_embedding": Tensor}`.
            senders: A tensor of shape (n_edges,) containing the sender node indices.
            receivers: A tensor of shape (n_edges,) containing the receiver node indices.
            n_node: A tensor of shape (1,) containing the number of nodes.
            atomic_numbers: A tensor of atomic numbers for reference energy calculation.

        Returns:
            ouput_dict: A dictionary containing the predictions:
            - `graph_pred`: Graph-level predictions (e.g., energy) of shape (n_graphs, graph_property_dim).
            - `stress_pred`: Stress predictions (if stress_head is provided) of shape (n_graphs, stress_dim).
            - `node_pred`: Node-level predictions of shape (n_nodes, node_property_dim).
        """
        _, nodes = self.model(edge_features, node_features, senders, receivers)

        output = {}
        output["graph_pred"] = self.graph_head.predict(nodes, n_node, atomic_numbers)
        output["stress_pred"] = self.stress_head.predict(nodes, n_node)
        output["node_pred"] = self.node_head.predict(nodes, n_node)

        return output

    def construct(self, edge_features, node_features, senders, receivers, n_node):
        """construct
        """
        edges, nodes = self.model(edge_features, node_features, senders, receivers)

        res = {"edges": edges, "nodes": nodes}
        res.update(self.graph_head(nodes, n_node))
        res.update(self.stress_head(nodes, n_node))
        res.update(self.node_head(nodes, n_node))

        return res
