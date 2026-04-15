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
"""Classes that define prediction targets."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import ase.data
import ase.db
import ase.db.row
import ase.db.sqlite
import numpy as np

import mindspore as ms
from mindspore import ops, Tensor, mint

HARTREE_TO_EV = 27.211386245988


def recursive_getattr(obj: object, attr: str) -> Any:
    """Recursively access an object property using dot notation."""
    for sub_attr in attr.split("."):
        obj = getattr(obj, sub_attr)

    return obj


def get_property_from_row(
        name: Union[str, List[str]],
        row: ase.db.row.AtomsRow,
        conversion_factor: float = 1.0,
) -> Tensor:
    """Retrieve arbitrary values from ase db data dict."""
    if isinstance(name, str):
        names = [name]
    else:
        names = name
    values = []
    for name_ in names:
        attribute = recursive_getattr(row, name_)
        target = np.array(attribute)
        values.append(target)

    property_tensor = ms.from_numpy(np.hstack(values)).to(ms.float32)

    while len(property_tensor.shape) < 2:
        property_tensor = property_tensor[None, ...]

    if "stress" in name and property_tensor.shape == (3, 3):
        # convert stress tensor to voigt notation
        property_tensor = Tensor(
            [
                property_tensor[0, 0],
                property_tensor[1, 1],
                property_tensor[2, 2],
                property_tensor[1, 2],
                property_tensor[0, 2],
                property_tensor[0, 1],
            ],
            dtype=ms.float32,
        ).unsqueeze(0)
    return property_tensor * conversion_factor


@dataclass
class PropertyDefinition:
    """Defines how to extract and transform a quantative property from an ase db.

    Such properties have two primary use-cases:
        - as features for the model to use / condition on.
        - as target variables for regression tasks.

    Args:
        name: The name of the property.
        dim: The dimensionality of the property variable.
        domain: Whether the variable is real, binary or categorical. If using
            this variable as a regression target, then var_type determines
            the loss function used e.g. MSE, BCE or cross-entropy loss.
        row_to_property_fn: A function defining how a target can be
            retrieved from an ase database row.
        means: The mean to transform this by in the model.
        stds: The std to scale this by in the model.
    """

    name: str
    dim: int
    domain: Literal["real", "binary", "categorical"]
    row_to_property_fn: Optional[Callable] = None
    means: Optional[Tensor] = None
    stds: Optional[Tensor] = None


def energy_row_fn(row: ase.db.row.AtomsRow, dataset: str) -> float:
    """Energy data in eV.

    - Some datasets use sums of energy values e.g. PBE + D3.
    - For external datasets, we should explicitly register how
      to extract the energy property by adding it to `extract_info'.
    - Unregistered datasets default to using the `energy` attribute
      and a conversion factor of 1, which is always correct for our
      internally generated datasets.
    """
    extract_info: Dict[str, List[Tuple]] = {
        "mp-traj": [("energy", 1)],
        "mp-traj-d3": [("energy", 1), ("data.d3.energy", 1)],
        "alexandria-d3": [("energy", 1), ("data.d3.energy", 1)],
    }
    if dataset not in extract_info:
        if not hasattr(row, "energy"):
            raise ValueError(
                f"db row {row.id} doesn't have an energy attribute directly "
                ", but also doesn't define a method to extract energy info."
            )
        return get_property_from_row("energy", row, 1)

    energy_ = 0.0
    for row_attribute, conversion_factor in extract_info[dataset]:
        energy_ += get_property_from_row(row_attribute, row, conversion_factor)
    return energy_


def forces_row_fn(row: ase.db.row.AtomsRow, dataset: str):
    """Force data in eV / Angstrom.

    - Some datasets use sums of energy values e.g. PBE + D3.
    - For external datasets, we should explicitly register how
      to extract the energy property by adding it to `extract_info'.
    - Unregistered datasets default to using the `energy` attribute
      and a conversion factor of 1, which is always correct for our
      internally generated datasets.
    """
    extract_info: Dict[str, List[Tuple]] = {
        "mp-traj": [("forces", 1)],
        "mp-traj-d3": [("forces", 1), ("data.d3.forces", 1)],
        "alexandria-d3": [("forces", 1), ("data.d3.forces", 1)],
    }
    if dataset not in extract_info:
        if not hasattr(row, "forces"):
            raise ValueError(
                f"db row {row.id} doesn't have a forces attribute directly, "
                "but also doesn't define a method to extract forces info."
            )
        return get_property_from_row("forces", row, 1)

    forces_ = 0.0
    for row_attribute, conversion_factor in extract_info[dataset]:
        forces_ += get_property_from_row(row_attribute, row, conversion_factor)
    return forces_


def stress_row_fn(row: ase.db.row.AtomsRow, dataset: str) -> float:
    """Extract stress data."""
    extract_info: Dict[str, List[Tuple]] = {
        "mp-traj": [("stress", 1)],
        "mp-traj-d3": [("stress", 1), ("data.d3.stress", 1)],
        "alexandria-d3": [("stress", 1), ("data.d3.stress", 1)],
    }
    if dataset not in extract_info:
        if not hasattr(row, "stress"):
            raise ValueError(
                f"db row {row.id} doesn't have an stress attribute directly "
                ", but also doesn't define a method to extract stress info."
            )
        return get_property_from_row("stress", row, 1)

    stress_ = 0.0
    for row_attribute, conversion_factor in extract_info[dataset]:
        stress_ += get_property_from_row(row_attribute, row, conversion_factor)
    return stress_


def test_fixture_node_row_fn(row: ase.db.row.AtomsRow):
    """Just return random noise."""

    pos = ms.from_numpy(row.toatoms().positions)
    return ops.rand_like(pos).to(ms.float32)


def test_fixture_graph_row_fn():
    """Just return random noise."""
    return mint.randn((1, 1)).to(ms.float32)


energy = PropertyDefinition(
    name="energy",
    dim=1,
    domain="real",
    row_to_property_fn=energy_row_fn,
)

forces = PropertyDefinition(
    name="forces",
    dim=3,
    domain="real",
    row_to_property_fn=forces_row_fn,
)

stress = PropertyDefinition(
    name="stress",
    dim=6,
    domain="real",
    row_to_property_fn=stress_row_fn,
)

test_fixture = PropertyDefinition(
    name="test-fixture",
    dim=3,
    domain="real",
    row_to_property_fn=test_fixture_node_row_fn,
)

test_graph_fixture = PropertyDefinition(
    name="test-graph-fixture",
    dim=1,
    domain="real",
    row_to_property_fn=test_fixture_graph_row_fn,
)


PROPERTIES = {
    "energy": energy,
    "forces": forces,
    "stress": stress,
    "test-fixture": test_fixture,
    "test-graph-fixture": test_graph_fixture,
}
