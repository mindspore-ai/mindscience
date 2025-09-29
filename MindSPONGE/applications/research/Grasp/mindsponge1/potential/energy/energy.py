# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
"""Base energy cell"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell

from ...function import functions as func
from ...function.units import Units


class EnergyCell(Cell):
    r"""
    Basic cell for energy term.

    Args:
        label (str):        Label (name) of energy.
        output_dim (int):   Output dimension. Default: 1
        length_unit (str):  Length unit for position coordinates. Default: 'nm'
        energy_unit (str):  Energy unit. Default: 'kj/mol'
        units (Units):      Units of length and energy. Default: None
        use_pbc (bool):     Whether to use periodic boundary condition. Default: None

    Returns:
        energy (Tensor), Tensor of shape (B, 1). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 label: str,
                 output_dim: int = 1,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 use_pbc: bool = None,
                 ):

        super().__init__()

        self.label = label
        self.output_dim = func.get_integer(output_dim)

        self.use_pbc = use_pbc

        if units is None:
            self.units = Units(length_unit, energy_unit)
        else:
            if not isinstance(units, Units):
                raise TypeError(
                    'The type of units must be "Unit" but get type: '+str(type(units)))
            self.units = units

        self.gather_values = func.gather_values
        self.gather_vectors = func.gather_vectors

        self.input_unit_scale = 1
        self.cutoff = None
        self.identity = ops.Identity()

    def set_input_unit(self, units: Units):
        """
        Set the length unit for the input coordinates.

        Args:
            units (Units):      Units of length and energy. Default: None.
        """
        if units is None:
            self.input_unit_scale = 1
        elif isinstance(units, Units):
            self.input_unit_scale = Tensor(
                self.units.convert_length_from(units), ms.float32)
        else:
            raise TypeError('Unsupported type: '+str(type(units)))
        return self

    def set_cutoff(self, cutoff: float):
        """
        Set cutoff distances.

        Args:
            cutoff (float):         Cutoff distance. Default: None.
        """
        if cutoff is None:
            self.cutoff = None
        else:
            self.cutoff = Tensor(cutoff, ms.float32)
        return self

    def set_pbc(self, use_pbc: bool = None):
        """
        Set whether to use periodic boundary condition.

        Args:
            use_pbc (bool, optional):     Whether to use periodic boundary condition. Default: None.
        """
        self.use_pbc = use_pbc
        return self

    def convert_energy_from(self, unit: str) -> float:
        """
        Convert energy from outside unit to inside unit.

        Args:
            unit (str):      Units of length and energy. Examples: 'nm', 'kj/mol'.

        Returns:
            float, energy from outside unit to inside unit.
        """
        return self.units.convert_energy_from(unit)

    def convert_energy_to(self, unit: str) -> float:
        """
        Convert energy from inside unit to outside unit.

        Args:
            unit (str):      Units of length and energy. Examples: 'nm', 'kj/mol'.

        Returns:
            float, energy from inside unit to outside unit.
        """
        return self.units.convert_energy_to(unit)

    @property
    def length_unit(self) -> float:
        return self.units.length_unit

    @property
    def energy_unit(self) -> float:
        return self.units.energy_unit

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""
        Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms. Default: None
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor), Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """
        raise NotImplementedError


class NonbondEnergy(EnergyCell):
    r"""
    Basic cell for non-bonded energy term

    Args:
        label (str):            Label (name) of energy.
        output_dim (int):       Dimension of the output. Default: 1
        cutoff (float):         cutoff distance. Default: None
        length_unit (str):      Length unit for position coordinates. Default: None
        energy_unit (str):      Energy unit. Default: None
        use_pbc (bool):         Whether to use periodic boundary condition. Default: None
        units (Units):          Units of length and energy. Default: None

    Returns:
        energy (Tensor), Tensor of shape (B, 1). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 label: str,
                 output_dim: int = 1,
                 cutoff: float = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 use_pbc: bool = None,
                 units: Units = None,
                 ):

        super().__init__(
            label=label,
            output_dim=output_dim,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
            use_pbc=use_pbc,
        )

        self.cutoff = None
        if cutoff is not None:
            self.cutoff = Tensor(cutoff, ms.float32)

        self.inverse_input_scale = 1

    def set_input_unit(self, units: Units):
        """
        Set the length unit for the input coordinates.

        Args:
            units (Units):          Units of length and energy. Default: None.
        """
        super().set_input_unit(units)
        self.inverse_input_scale = msnp.reciprocal(self.input_unit_scale)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  pbc_box: Tensor = None,
                  ):
        r"""Calculate energy term

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor), Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """

        raise NotImplementedError
