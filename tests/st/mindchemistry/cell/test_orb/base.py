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
"""Base data class."""

from typing import Dict, Mapping, NamedTuple, Optional, Union

from mindspore import Tensor


Metric = Union[Tensor, int, float]
TensorDict = Mapping[str, Optional[Tensor]]


class ModelOutput(NamedTuple):
    """A model's output."""

    loss: Tensor
    log: Mapping[str, Metric]


class AtomGraphs(NamedTuple):
    """A class representing the input to a model for a graph.

    Args:
        senders (ms.Tensor): The integer source nodes for each edge.
        receivers (ms.Tensor): The integer destination nodes for each edge.
        n_node (ms.Tensor): A (batch_size, ) shaped tensor containing the number of nodes per graph.
        n_edge (ms.Tensor): A (batch_size, ) shaped tensor containing the number of edges per graph.
        node_features (Dict[str, ms.Tensor]): A dictionary containing node feature tensors.
            It will always contain "atomic_numbers" and "positions" keys, representing the
            atomic numbers of each node, and the 3d cartesian positions of them respectively.
        edge_features (Dict[str, ms.Tensor]): A dictionary containing edge feature tensors.
        system_features (Optional[TensorDict]): An optional dictionary containing system-level features.
        node_targets (Optional[Dict[ms.Tensor]]): An optional dict of tensors containing targets
            for individual nodes. This tensor is commonly expected to have shape (num_nodes, *).
        edge_target (Optional[ms.Tensor]): An optional tensor containing targets for individual edges.
            This tensor is commonly expected to have (num_edges, *).
        system_targets (Optional[Dict[ms.Tensor]]): An optional dict of tensors containing targets for the
            entire system. system_id (Optional[ms.Tensor]): An optional tensor containing the ID of the system.
        fix_atoms (Optional[ms.Tensor]): An optional tensor containing information on fixed atoms in the system.
    """

    senders: Tensor
    receivers: Tensor
    n_node: Tensor
    n_edge: Tensor
    node_features: Dict[str, Tensor]
    edge_features: Dict[str, Tensor]
    system_features: Dict[str, Tensor]
    node_targets: Optional[Dict[str, Tensor]] = None
    edge_targets: Optional[Dict[str, Tensor]] = None
    system_targets: Optional[Dict[str, Tensor]] = None
    system_id: Optional[Tensor] = None
    fix_atoms: Optional[Tensor] = None
    tags: Optional[Tensor] = None
    radius: Optional[float] = None
    max_num_neighbors: Optional[int] = None

    @property
    def positions(self):
        """Get positions of atoms."""
        return self.node_features["positions"]

    @positions.setter
    def positions(self, val: Tensor):
        self.node_features["positions"] = val

    @property
    def atomic_numbers(self):
        """Get integer atomic numbers."""
        return self.node_features["atomic_numbers"]

    @atomic_numbers.setter
    def atomic_numbers(self, val: Tensor):
        self.node_features["atomic_numbers"] = val

    @property
    def cell(self):
        """Get unit cells."""
        assert self.system_features
        return self.system_features.get("cell")

    @cell.setter
    def cell(self, val: Tensor):
        assert self.system_features
        self.system_features["cell"] = val

    def clone(self):
        """Clone the AtomGraphs object."""
        return AtomGraphs(
            senders=self.senders.clone(),
            receivers=self.receivers.clone(),
            n_node=self.n_node.clone(),
            n_edge=self.n_edge.clone(),
            node_features={k: v.clone() for k, v in self.node_features.items()},
            edge_features={k: v.clone() for k, v in self.edge_features.items()},
            system_features={k: v.clone() for k, v in self.system_features.items()},
            node_targets={k: v.clone() for k, v in (self.node_targets or {}).items()},
            edge_targets=self.edge_targets.clone() if self.edge_targets is not None else None,
            system_targets={k: v.clone() for k, v in (self.system_targets or {}).items()},
            system_id=self.system_id.clone() if self.system_id is not None else None,
            fix_atoms=self.fix_atoms.clone() if self.fix_atoms is not None else None,
            tags=self.tags.clone() if self.tags is not None else None,
            radius=self.radius,
            max_num_neighbors=self.max_num_neighbors
        )
