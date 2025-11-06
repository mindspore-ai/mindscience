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
"""Base Model class."""

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Union
import tree

import mindspore as ms
from mindspore import ops, Tensor, mint

from src import featurization_utilities

Metric = Union[Tensor, int, float]
TensorDict = Mapping[str, Optional[Tensor]]


class ModelOutput(NamedTuple):
    """A model's output."""

    loss: Tensor
    log: Mapping[str, Metric]


class AtomGraphs(NamedTuple):
    """A class representing the input to a model for a graph.

    Args:
        senders (torch.Tensor): The integer source nodes for each edge.
        receivers (torch.Tensor): The integer destination nodes for each edge.
        n_node (torch.Tensor): A (batch_size, ) shaped tensor containing the number of nodes per graph.
        n_edge (torch.Tensor): A (batch_size, ) shaped tensor containing the number of edges per graph.
        node_features (Dict[str, torch.Tensor]): A dictionary containing node feature tensors.
            It will always contain "atomic_numbers" and "positions" keys, representing the
            atomic numbers of each node, and the 3d cartesian positions of them respectively.
        edge_features (Dict[str, torch.Tensor]): A dictionary containing edge feature tensors.
        system_features (Optional[TensorDict]): An optional dictionary containing system-level features.
        node_targets (Optional[Dict[torch.Tensor]]): An optional dict of tensors containing targets
            for individual nodes. This tensor is commonly expected to have shape (num_nodes, *).
        edge_target (Optional[torch.Tensor]): An optional tensor containing targets for individual edges.
            This tensor is commonly expected to have (num_edges, *).
        system_targets (Optional[Dict[torch.Tensor]]): An optional dict of tensors containing targets for the
            entire system. system_id (Optional[torch.Tensor]): An optional tensor containing the ID of the system.
        fix_atoms (Optional[torch.Tensor]): An optional tensor containing information on fixed atoms in the system.
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

    def clone(self) -> "AtomGraphs":
        """Clone the AtomGraphs object.

        Note: this differs from deepcopy() because it preserves gradients.
        """

        def _clone(x):
            if isinstance(x, Tensor):
                return x.clone()
            return x

        return tree.map_structure(_clone, self)

    def to(self, device: Any = None) -> "AtomGraphs":
        """Move AtomGraphs child tensors to a device."""

        print(f"Moving AtomGraphs to device: {device}")
        def _to(x):
            if hasattr(x, "to"):
                return x
            return x

        return tree.map_structure(_to, self)

    def tachdetach(self) -> "AtomGraphs":
        """Detach all child tensors."""

        def _detach(x):
            if hasattr(x, "detach"):
                return x.detach()
            return x

        return tree.map_structure(_detach, self)

    def equals(self, graphs: "AtomGraphs") -> bool:
        """Check two atomgraphs are equal."""

        def _is_equal(x, y):
            if isinstance(x, Tensor):
                return mint.equal(x, y)
            return x == y

        flat_results = tree.flatten(tree.map_structure(_is_equal, self, graphs))
        return all(flat_results)

    def allclose(self, graphs: "AtomGraphs", rtol=1e-5, atol=1e-8) -> bool:
        """Check all tensors/scalars of two atomgraphs are close."""

        def _is_close(x, y):
            if isinstance(x, Tensor):
                return mint.allclose(x, y, rtol=rtol, atol=atol)
            if isinstance(x, (float, int)):
                return mint.allclose(
                    Tensor(x), Tensor(y), rtol=rtol, atol=atol
                )
            return x == y

        flat_results = tree.flatten(tree.map_structure(_is_close, self, graphs))
        return all(flat_results)

    def to_dict(self):
        """Return a dictionary mapping each AtomGraph property to a corresponding tensor/scalar.

        Any nested attributes of the AtomGraphs are unpacked so the
        returned dict has keys like "positions" and "atomic_numbers".

        Any None attributes are not included in the dictionary.

        Returns:
            dict: A dictionary mapping attribute_name -> tensor/scalar
        """
        ret = {}
        for key, val in self._asdict().items():
            if val is None:
                continue
            if isinstance(val, dict):
                for k, v in val.items():
                    ret[k] = v
            else:
                ret[key] = val

        return ret

    def to_batch_dict(self) -> Dict[str, Any]:
        """Return a single dictionary mapping each AtomGraph property to a corresponding list of tensors/scalars.

        Returns:
            dict: A dict mapping attribute_name -> list of length batch_size containing tensors/scalars.
        """
        batch_dict = defaultdict(list)
        for graph in self.split(self):
            for key, value in graph.to_dict().items():
                batch_dict[key].append(value)
        return batch_dict

    def split(self, clone=True) -> List["AtomGraphs"]:
        """Splits batched AtomGraphs into constituent system AtomGraphs.

        Args:
            graphs (AtomGraphs): A batched AtomGraphs object.
            clone (bool): Whether to clone the graphs before splitting.
                Cloning removes risk of side-effects, but uses more memory.
        """
        graphs = self.clone() if clone else self

        batch_nodes = graphs.n_node.tolist()
        batch_edges = graphs.n_edge.tolist()

        if not batch_nodes:
            raise ValueError("Cannot split empty batch")
        if len(batch_nodes) == 1:
            return [graphs]

        batch_systems = mint.ones(len(batch_nodes), dtype=ms.int32).tolist()
        node_features = _split_features(graphs.node_features, batch_nodes)
        node_targets = _split_features(graphs.node_targets, batch_nodes)
        edge_features = _split_features(graphs.edge_features, batch_edges)
        edge_targets = _split_features(graphs.edge_targets, batch_edges)
        system_features = _split_features(graphs.system_features, batch_systems)
        system_targets = _split_features(graphs.system_targets, batch_systems)
        system_ids = _split_tensors(graphs.system_id, batch_systems)
        fix_atoms = _split_tensors(graphs.fix_atoms, batch_nodes)
        tags = _split_tensors(graphs.tags, batch_nodes)
        batch_nodes = [Tensor([n]) for n in batch_nodes]
        batch_edges = [Tensor([e]) for e in batch_edges]

        # calculate the new senders and receivers
        senders = list(_split_tensors(graphs.senders, batch_edges))
        receivers = list(_split_tensors(graphs.receivers, batch_edges))
        n_graphs = graphs.n_node.shape[0]
        offsets = mint.cumsum(graphs.n_node[:-1], 0)
        offsets = mint.cat([Tensor([0]), offsets])
        unbatched_senders = []
        unbatched_recievers = []
        for graph_index in range(n_graphs):
            s = senders[graph_index] - offsets[graph_index]
            r = receivers[graph_index] - offsets[graph_index]
            unbatched_senders.append(s)
            unbatched_recievers.append(r)

        return [
            AtomGraphs(*args)
            for args in zip(
                unbatched_senders,
                unbatched_recievers,
                batch_nodes,
                batch_edges,
                node_features,
                edge_features,
                system_features,
                node_targets,
                edge_targets,
                system_targets,
                system_ids,
                fix_atoms,
                tags,
                [graphs.radius for _ in range(len(batch_nodes))],
                [graphs.max_num_neighbors for _ in range(len(batch_nodes))],
            )
        ]


def batch_graphs(graphs: List[AtomGraphs]) -> AtomGraphs:
    """Batch graphs together by concatenating their nodes, edges, and features.

    Args:
        graphs (List[AtomGraphs]): A list of AtomGraphs to be batched together.

    Returns:
        AtomGraphs: A new AtomGraphs object with the concatenated nodes,
        edges, and features from the input graphs, along with concatenated target,
        system ID, and other information.
    """
    # Calculates offsets for sender and receiver arrays, caused by concatenating
    # the nodes arrays.
    offsets = mint.cumsum(
        Tensor([0] + [mint.sum(g.n_node) for g in graphs[:-1]]), 0
    )
    radius = graphs[0].radius
    assert {graph.radius for graph in graphs} == {radius}
    max_num_neighbours = graphs[0].max_num_neighbors
    assert {graph.max_num_neighbors for graph in graphs} == {max_num_neighbours}

    return AtomGraphs(
        n_node=mint.concat([g.n_node for g in graphs], dim=0).to(ms.int64),
        n_edge=mint.concat([g.n_edge for g in graphs], dim=0).to(ms.int64),
        senders=mint.concat(
            [g.senders + o for g, o in zip(graphs, offsets)], dim=0
        ).to(ms.int64),
        receivers=mint.concat(
            [g.receivers + o for g, o in zip(graphs, offsets)], dim=0
        ).to(ms.int64),
        node_features=_map_concat([g.node_features for g in graphs]),
        edge_features=_map_concat([g.edge_features for g in graphs]),
        system_features=_map_concat([g.system_features for g in graphs]),
        node_targets=_map_concat([g.node_targets for g in graphs]),
        edge_targets=_map_concat([g.edge_targets for g in graphs]),
        system_targets=_map_concat([g.system_targets for g in graphs]),
        system_id=_concat([g.system_id for g in graphs]),
        fix_atoms=_concat([g.fix_atoms for g in graphs]),
        tags=_concat([g.tags for g in graphs]),
        radius=radius,
        max_num_neighbors=max_num_neighbours,
    )


def refeaturize_atomgraphs(
        atoms: AtomGraphs,
        positions: Tensor,
        atomic_number_embeddings: Optional[Tensor] = None,
        cell: Optional[Tensor] = None,
        recompute_neighbors=True,
        updates: Optional[Tensor] = None,
        fixed_atom_pos: Optional[Tensor] = None,
        fixed_atom_type_embedding: Optional[Tensor] = None,
        differentiable: bool = False,
) -> AtomGraphs:
    """Return a graph updated according to the new positions, and (if given) atomic numbers and unit cells.

    Note: if a unit cell is given, it will *both* be used to do the
    pbc-remapping and be set on the returned AtomGraphs

    Args:
        atoms (AtomGraphs): The original AtomGraphs object.
        positions (torch.Tensor): The new positions of the atoms.
        atomic_number_embeddings (Optional[torch.Tensor]): The new atomic number embeddings.
        cell (Optional[torch.Tensor]): The new unit cell.
        recompute_neighbors (bool): Whether to recompute the neighbor list.
        updates (Optional[torch.Tensor]): The updates to the positions.
        fixed_atom_pos (Optional[torch.Tensor]): The positions of atoms
            which are fixed when diffusing on a fixed trajectory.
        fixed_atom_type_embedding (Optional[torch.Tensor]): If using atom type diffusion
            with a fixed trajectory, the unormalized vectors of the fixed atoms. Shape (n_atoms, 118).
        differentiable (bool): Whether to make the graph inputs require_grad. This includes
            the positions and atomic number embeddings, if passed.
        exact_pbc_image_neighborhood: bool: If the exact pbc image neighborhood calculation (from torch nl)
            which considers boundary crossing for more than cell is used.

    Returns:
        AtomGraphs: A refeaturized AtomGraphs object.
    """
    if cell is None:
        cell = atoms.cell
    if atoms.fix_atoms is not None and fixed_atom_pos is not None:
        positions[atoms.fix_atoms] = fixed_atom_pos[atoms.fix_atoms]

    if (
            atoms.fix_atoms is not None
            and fixed_atom_type_embedding is not None
            and atomic_number_embeddings is not None
    ):
        atomic_number_embeddings[atoms.fix_atoms] = fixed_atom_type_embedding[atoms.fix_atoms]

    num_atoms = atoms.n_node
    positions = featurization_utilities.batch_map_to_pbc_cell(positions, cell, num_atoms)

    if differentiable:
        positions.requires_grad = True
        if atomic_number_embeddings is not None:
            atomic_number_embeddings.requires_grad = True

    if recompute_neighbors:
        assert atoms.radius is not None and atoms.max_num_neighbors is not None
        (
            edge_index,
            edge_vectors,
            batch_num_edges,
        ) = featurization_utilities.batch_compute_pbc_radius_graph(
            positions=positions,
            periodic_boundaries=cell,
            radius=atoms.radius,
            image_idx=num_atoms,
            max_number_neighbors=atoms.max_num_neighbors,
        )
        new_senders = edge_index[0]
        new_receivers = edge_index[1]
    else:
        assert updates is not None
        new_senders = atoms.senders
        new_receivers = atoms.receivers
        edge_vectors = recompute_edge_vectors(atoms, updates)
        batch_num_edges = atoms.n_edge

    edge_features = {"vectors": edge_vectors.to(ms.float32),}

    new_node_features = {}
    if atoms.node_features is not None:
        new_node_features = deepcopy(atoms.node_features)
    new_node_features["positions"] = positions
    if atomic_number_embeddings is not None:
        new_node_features["atomic_numbers_embedding"] = atomic_number_embeddings

    new_system_features = {}
    if atoms.system_features is not None:
        new_system_features = deepcopy(atoms.system_features)
    new_system_features["cell"] = cell

    new_atoms = AtomGraphs(
        senders=new_senders,
        receivers=new_receivers,
        n_node=atoms.n_node,
        n_edge=batch_num_edges,
        node_features=new_node_features,
        edge_features=edge_features,
        system_features=new_system_features,
        node_targets=atoms.node_targets,
        system_targets=atoms.system_targets,
        fix_atoms=atoms.fix_atoms,
        tags=atoms.tags,
        radius=atoms.radius,
        max_num_neighbors=atoms.max_num_neighbors,
    )
    return new_atoms


def recompute_edge_vectors(atoms, updates):
    """Recomputes edge vectors with per node updates."""
    updates = -updates
    senders = atoms.senders
    receivers = atoms.receivers
    edge_translation = updates[senders] - updates[receivers]
    return atoms.edge_features["vectors"] + edge_translation


def volume_atomgraphs(atoms: AtomGraphs):
    """Returns the volume of the unit cell."""
    cell = atoms.cell
    cross = ops.Cross(dim=1)
    return (cell[:, 0] * cross(cell[:, 1], cell[:, 2])).sum(-1)


def _concat_tensors(*args):
    """Concatenates tensors."""
    return _concat(args)


def _map_concat(nests):
    """Maps and concatenates nested structures."""
    return tree.map_structure(_concat_tensors, *nests)


def _concat(
        tensors: List[Optional[Tensor]],
) -> Optional[Tensor]:
    """Splits tensors based on the intended split sizes."""
    if any(x is None for x in tensors):
        return None
    return mint.concat(tensors, dim=0)


def _split_tensors(
        features: Optional[Tensor],
        split_sizes: List[int],
) -> Sequence[Optional[Tensor]]:
    """Splits tensors based on the intended split sizes."""
    if features is None:
        return [None] * len(split_sizes)

    return mint.split(features, split_sizes)


def _split_features(
        features: Optional[TensorDict],
        split_sizes: List[int],
) -> Sequence[Optional[TensorDict]]:
    """Splits features based on the intended split sizes."""
    if features is None:
        return [None] * len(split_sizes)

    split_dict = {
        k: mint.split(v, split_sizes) if v is not None else [None] * len(split_sizes)
        for k, v in features.items()
    }
    individual_tuples = zip(*list(split_dict.values()))
    individual_dicts: List[Optional[TensorDict]] = list(
        map(lambda k: dict(zip(split_dict.keys(), k)), individual_tuples)
    )
    return individual_dicts
