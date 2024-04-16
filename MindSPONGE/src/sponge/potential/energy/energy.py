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
"""Base energy cell"""

from typing import Union
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell

from ...system.molecule import Molecule
from ...function import get_ms_array
from ...function.units import Units, Length, GLOBAL_UNITS

_ENERGY_BY_KEY = dict()


def _energy_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _ENERGY_BY_KEY:
            _ENERGY_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _ENERGY_BY_KEY:
                _ENERGY_BY_KEY[alias] = cls

        return cls

    return alias_reg


class EnergyCell(Cell):
    r"""
    Base class for energy terms.
    `EnergyCell` is usually used as a base class for individual energy terms in a classical force field.
    As the force field parameters usually has units, the units of the `EnergyCell` as an energy term
    should be the same as the units of the force field parameters, and not equal to the global units.

    Note:
        B:  Batchsize, i.e. number of walkers in simulation

    Args:
        name (str):         Name of energy. Default: 'energy'
        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: 'nm'
        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: 'kj/mol'
        use_pbc (bool):     Whether to use periodic boundary condition. Default: ``None``.
        kwargs (dict):      Other parameters dictionary.

    Returns:
        Tensor of energy, Tensor of shape `(B, 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore.nn import Adam
        >>> from sponge.potential import EnergyCell, ForceFieldBase
        >>> from sponge import WithEnergyCell, Sponge
        >>> from sponge.callback import RunInfo
        >>> class MyEnergy(EnergyCell):
        ...     def construct(self, coordinate: Tensor, **kwargs):
        ...         return coordinate.sum()[None, None]
        >>> # system represents a custom molecular system
        >>> potential = MyEnergy(system)
        >>> forcefield = ForceFieldBase(potential)
        >>> withenergy = WithEnergyCell(system, forcefield)
        >>> opt = Adam(system.trainable_params(), 1e-3)
        >>> mini = Sponge(withenergy, optimizer=opt)
        >>> run_info = RunInfo(5)
        >>> mini.run(10, callbacks=[run_info])
        [MindSPONGE] Started simulation at 2024-03-22 11:08:34
        [MindSPONGE] Step: 5, E_pot: 0.31788814
        [MindSPONGE] Step: 10, E_pot: 0.13788882
        [MindSPONGE] Finished simulation at 2024-03-22 11:08:35
        [MindSPONGE] Simulation time: 0.98 seconds.
        --------------------------------------------------------------------------------
    """
    def __init__(self,
                 name: str = 'energy',
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 use_pbc: bool = None,
                 **kwargs
                 ):
        super().__init__()
        self._kwargs = kwargs

        self._name = name

        self._use_pbc = use_pbc

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        if energy_unit is None:
            energy_unit = GLOBAL_UNITS.energy_unit
        self.units = Units(length_unit, energy_unit)

        self.input_unit_scale = 1
        self.cutoff = None
        self.identity = ops.Identity()

    @property
    def name(self) -> str:
        """
        Name of energy.

        Returns:
            str, name of energy.
        """
        return self._name

    @property
    def use_pbc(self) -> bool:
        """
        Whether to use periodic boundary condition.

        Returns:
            bool, the flag used to judge whether to use periodic boundary condition.
        """
        return self._use_pbc

    @property
    def length_unit(self) -> str:
        """
        Length unit.

        Returns:
            str, length unit.
        """
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        """
        Energy unit.

        Returns:
            str, energy unit.
        """
        return self.units.energy_unit

    @staticmethod
    def check_system(system: Molecule) -> bool:
        """
        Check if the system needs to calculate this energy term

        Args:
            system (Molecule): System.
        """
        #pylint:disable=unused-argument
        return True

    def set_units(self, length_unit: str = None, energy_unit: str = None, units: Units = None):
        r"""
        Set length and energy units.

        Args:
            length_unit (str):  Length unit. Only valid when `units` is ``None`` .
                                Default: ``None``
            energy_unit (str):  Energy unit. Only valid when `units` is ``None`` .
                                Default: ``None``
            units (Units):      `Units` object. Default: ``None``
        """
        if units is None:
            if length_unit is None:
                length_unit = GLOBAL_UNITS.length_unit
            if energy_unit is None:
                energy_unit = GLOBAL_UNITS.energy_unit
        else:
            length_unit = None
            energy_unit = None

        if self.units is None:
            self.units = Units(length_unit=length_unit, energy_unit=energy_unit, units=units)
        else:
            self.units.set_units(length_unit=length_unit, energy_unit=energy_unit, units=units)

        return self

    def set_input_unit(self, length_unit: Union[str, Units, Length]):
        """
        Set the length unit for the input coordinates.

        Args:
            length_unit(Union[str, Units, Length]): The length unit for the input coordinates.
        """
        if length_unit is None:
            self.input_unit_scale = 1
        elif isinstance(length_unit, (str, Units, float)):
            self.input_unit_scale = Tensor(
                self.units.convert_length_from(length_unit), ms.float32)
        else:
            raise TypeError(f'Unsupported type of `length_unit`: {type(length_unit)}')

        return self

    def set_cutoff(self, cutoff: float, unit: str = None):
        """
        Set cutoff distances.

        Args:
            cutoff(float):  Cutoff distances.
            unit(str):      Length unit. Default: ``None``.
        """
        if cutoff is None:
            self.cutoff = None
        else:
            cutoff = get_ms_array(cutoff, ms.float32)
            self.cutoff = self.units.length(cutoff, unit)
        return self

    def set_pbc(self, use_pbc: bool):
        """
        Set whether to use periodic boundary condition.

        Args:
            use_pbc(bool): Whether to use periodic boundary condition.
        """
        self._use_pbc = use_pbc
        return self

    def convert_energy_from(self, unit: str) -> float:
        """
        Convert energy from outside unit to inside unit.

        Args:
            unit(str):  Energy unit.

        Returns:
            float, energy according from a specified units.
        """
        return self.units.convert_energy_from(unit)

    def convert_energy_to(self, unit: str) -> float:
        """
        Convert energy from inside unit to outside unit.

        Args:
            unit(str):  Energy unit.

        Returns:
            float, energy according to a specified units.
        """
        return self.units.convert_energy_to(unit)

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: ``None``.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index. Default: ``None``.
            neighbour_vector (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Vectors from central atom to neighbouring atoms. Default: ``None``.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: ``None``.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: ``None``.

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        raise NotImplementedError


class NonbondEnergy(EnergyCell):
    r"""Base cell for non-bonded energy terms.

    Args:
        name (str):             Name of energy.

        cutoff (Union[float, Length, Tensor]):  cutoff distance. Default: ``None``.

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: 'nm'

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: 'kj/mol'

        use_pbc (bool):         Whether to use periodic boundary condition. Default: ``None``.

    """
    def __init__(self,
                 name: str,
                 cutoff: Union[float, Length, Tensor] = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 use_pbc: bool = None,
                 ):

        super().__init__(
            name=name,
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
        )

        if isinstance(cutoff, Length):
            cutoff = cutoff(self.units)

        self.cutoff = None
        if cutoff is not None:
            self.cutoff = Tensor(cutoff, ms.float32)

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: ``None``.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index. Default: ``None``.
            neighbour_vector (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Vectors from central atom to neighbouring atoms. Default: ``None``.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: ``None``.
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances. Default: ``None``.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: ``None``.

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        raise NotImplementedError
