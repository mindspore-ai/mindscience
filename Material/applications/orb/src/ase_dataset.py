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
"""ASE dataset"""

import os
from typing import Dict, Optional, Tuple, Union

import ase
import ase.db
import ase.db.row
import ase.stress
import numpy as np
import mindspore as ms
from mindspore import Tensor

from src import atomic_system, property_definitions
from src.base import AtomGraphs
from src.utils import rand_matrix


class AseSqliteDataset:
    """AseSqliteDataset.

    A MindSpore Dataset for reading ASE Sqlite serialized Atoms objects.

    Args:
        dataset_path: Local path to read.
        system_config: A config for controlling how an atomic system is represented.
        target_config: A config for regression/classification targets.
        augmentation: If random rotation augmentation is used.

    Returns:
        An AseSqliteDataset.
    """

    def __init__(
            self,
            dataset_path: Union[str, os.PathLike],
            system_config: Optional[atomic_system.SystemConfig] = None,
            target_config: Optional[Dict] = None,
            augmentation: Optional[bool] = True,
    ):
        super().__init__()
        self.augmentation = augmentation
        self.path = dataset_path
        self.db = ase.db.connect(str(self.path), serial=True, type="db")

        self.feature_config = system_config
        if target_config is None:
            target_config = {
                "graph": ["energy", "stress"],
                "node": ["forces"],
                "edge": [],
            }
        self.target_config = target_config

    def __getitem__(self, idx) -> AtomGraphs:
        """Fetch an item from the db.

        Args:
            idx: An index to fetch from the db file and convert to an AtomGraphs.

        Returns:
            A AtomGraphs object containing everything the model needs as input,
            positions and atom types and other auxiliary information, such as
            fine tuning targets, or global graph features.
        """
        # Sqlite db is 1 indexed.
        row = self.db.get(idx + 1)
        atoms = row.toatoms()
        node_properties = property_definitions.get_property_from_row(
            self.target_config["node"], row
        )
        graph_property_dict = {}
        for target_property in self.target_config["graph"]:
            system_properties = property_definitions.get_property_from_row(
                target_property, row
            )
            # transform stress to voigt6 representation
            if target_property == "stress" and len(system_properties.reshape(-1)) == 9:
                system_properties = Tensor(
                    ase.stress.full_3x3_to_voigt_6_stress(system_properties.reshape(3, 3)),
                    dtype=ms.float32,
                ).reshape(1, -1)
            graph_property_dict[target_property] = system_properties
        extra_targets = {
            "node": {"forces": node_properties},
            "edge": {},
            "graph": graph_property_dict,
        }
        if self.augmentation:
            atoms, extra_targets = random_rotations_with_properties(atoms, extra_targets)

        atom_graph = atomic_system.ase_atoms_to_atom_graphs(
            atoms,
            system_id=idx,
            brute_force_knn=False,
        )
        atom_graph = self._add_extra_targets(atom_graph, extra_targets)

        return atom_graph

    def get_atom(self, idx: int) -> ase.Atoms:
        """Return the Atoms object for the dataset index."""
        row = self.db.get(idx + 1)
        return row.toatoms()

    def get_atom_and_metadata(self, idx: int) -> Tuple[ase.Atoms, Dict]:
        """Return the Atoms object plus a dict of metadata for the dataset index."""
        row = self.db.get(idx + 1)
        return row.toatoms(), row.data

    def __len__(self) -> int:
        """Return the dataset length."""
        return len(self.db)

    def __repr__(self) -> str:
        """String representation of class."""
        return f"AseSqliteDataset(path={self.path})"

    def _add_extra_targets(
            self,
            atom_graph: AtomGraphs,
            extra_targets: Dict[str, Dict],
    ):
        """Add extra features and targets to the AtomGraphs object.

        Args:
            atom_graph: AtomGraphs object to add extra features and targets to.
            extra_targets: Dictionary of extra targets to add.
        """
        node_targets = (
            atom_graph.node_targets if atom_graph.node_targets is not None else {}
        )
        node_targets = {**node_targets, **extra_targets["node"]}

        edge_targets = (
            atom_graph.edge_targets if atom_graph.edge_targets is not None else {}
        )
        edge_targets = {**edge_targets, **extra_targets["edge"]}

        system_targets = (
            atom_graph.system_targets if atom_graph.system_targets is not None else {}
        )
        system_targets = {**system_targets, **extra_targets["graph"]}

        return atom_graph._replace(
            node_targets=node_targets if node_targets != {} else None,
            edge_targets=edge_targets if edge_targets != {} else None,
            system_targets=system_targets if system_targets != {} else None,
        )


def random_rotations_with_properties(
        atoms: ase.Atoms, properties: dict
) -> Tuple[ase.Atoms, dict]:
    """Randomly rotate atoms in ase.Atoms object.

    This exists to handle the case where we also need to rotate properties.
    Currently we only ever do this for random rotations, but it could be extended.

    Args:
        atoms (ase.Atoms): Atoms object to rotate.
        properties (dict): Dictionary of properties to rotate.
    """
    rand_rotation = rand_matrix(1)[0]
    atoms.positions = atoms.positions @ rand_rotation
    if atoms.cell is not None:
        atoms.set_cell(atoms.cell.array @ rand_rotation)

    new_node_properties = {}
    for key, v in properties["node"].items():
        if tuple(v.shape) == tuple(atoms.positions.shape):
            new_node_properties[key] = v @ rand_rotation
        else:
            new_node_properties[key] = v
    properties["node"] = new_node_properties

    if "stress" in properties["graph"]:
        # Transformation rule of stress tensor
        stress = properties["graph"]["stress"]
        full_stress = ase.stress.voigt_6_to_full_3x3_stress(stress)

        if full_stress.shape != (3, 3):
            full_stress = full_stress.reshape(3, 3)

        transformed = np.dot(np.dot(rand_rotation, full_stress), rand_rotation.T)
        # Back to voigt notation, and shape (1, 6) for consistency with batching
        properties["graph"]["stress"] = Tensor(
            [
                transformed[0, 0],
                transformed[1, 1],
                transformed[2, 2],
                transformed[1, 2],
                transformed[0, 2],
                transformed[0, 1],
            ],
            dtype=ms.float32,
        ).unsqueeze(0)

    return atoms, properties

class BufferData:
    """Wrapper for a dataset. Loads all data into memory."""

    def __init__(self, dataset, shuffle: bool = True):
        """BufferData.
        Args:
            dataset: The dataset to wrap.
            shuffle: If True, shuffle the data.
        """
        self.data_objects = [dataset[i] for i in range(len(dataset))]
        if shuffle:
            self.shuffle()

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, index):
        return self.data_objects[index]

    def shuffle(self):
        """Shuffle the data."""
        indices = np.arange(len(self.data_objects))
        np.random.shuffle(indices)
        self.data_objects = [self.data_objects[i] for i in indices]
