# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
Energy terms with `EnergCell` as base class
"""

from typing import Union
from mindspore import Tensor
from ...system import Molecule

from .energy import EnergyCell, NonbondEnergy, _ENERGY_BY_KEY
from .bond import BondEnergy
from .angle import AngleEnergy
from .dihedral import DihedralEnergy
from .improper import ImproperEnergy
from .cmap import CmapEnergy
from .coulomb import CoulombEnergy
from .lj import LennardJonesEnergy
from .pairs import NonbondPairwiseEnergy

__all__ = ['EnergyCell', 'NonbondEnergy', 'BondEnergy', 'AngleEnergy', 'DihedralEnergy',
           'ImproperEnergy', 'CoulombEnergy', 'LennardJonesEnergy', 'NonbondPairwiseEnergy',
           'get_energy_cell', 'CmapEnergy']

_ENERGY_BY_NAME = {cell.__name__: cell for cell in _ENERGY_BY_KEY.values()}


def get_energy_cell(cls_name: Union[str, dict, EnergyCell],
                    system: Molecule,
                    parameters: dict = None,
                    exclude_index: Tensor = None,
                    **kwargs) -> EnergyCell:
    r"""get object of energy cell

    Args:
        cls_name (Union[str, dict, Thermostat]): Class name, arguments or object of a energy cell.
        system (Molecule): Simulation system.
        parameters (dict): Dict of force field parameters. Default: None
        exclude_index (Tensor): Tensor of exclude index for neighbour list. Default: None
        **kwargs: Other arguments

    Return:
        energy (EnergyCell): Object of energy cell

    """

    if cls_name is None:
        return None

    if isinstance(cls_name, EnergyCell):
        return cls_name

    if isinstance(cls_name, dict):
        return get_energy_cell(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None

        #pylint: disable=invalid-name
        if cls_name.lower() in _ENERGY_BY_KEY:
            EnergyCell_: EnergyCell = _ENERGY_BY_KEY.get(cls_name.lower())
        elif cls_name in _ENERGY_BY_NAME:
            EnergyCell_: EnergyCell = _ENERGY_BY_NAME.get(cls_name.lower())
        else:
            raise ValueError(f"The energy cell corresponding to '{cls_name}' was not found.")

        if EnergyCell_.check_system(system):
            return EnergyCell_(system=system,
                               parameters=parameters,
                               exclude_index=exclude_index,
                               **kwargs)
        return None

    raise TypeError(f"Unsupported energy cell type '{type(cls_name)}'.")
