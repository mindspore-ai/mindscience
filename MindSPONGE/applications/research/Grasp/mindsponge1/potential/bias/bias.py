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
"""Base cell for bais potential"""

from mindspore import Tensor

from ..potential import PotentialCell
from ...colvar import Colvar
from ...function.units import Units, global_units


class Bias(PotentialCell):
    r"""
    Basic cell for bias potential.

    Args:
        colvar (Colvar):            Collective variables. Default: None
        multiple_walkers (bool):    Whether to use multiple walkers. Default: False
        length_unit (str):          Length unit for position coordinates. Default: None
        energy_unit (str):          Energy unit. Default: None
        units (Units):              Units of length and energy. Default: global_units
        use_pbc (bool):             Whether to use periodic boundary condition. Default: None

    Returns:
        potential (Tensor), Tensor of shape (B, 1). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 colvar: Colvar = None,
                 multiple_walkers: bool = False,
                 length_unit: str = None,
                 energy_unit: str = None,
                 units: Units = global_units,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
            use_pbc=use_pbc,
        )

        if units is None:
            self.units.set_length_unit(length_unit)
            self.units.set_energy_unit(energy_unit)
        else:
            self.units = units

        self.colvar = colvar
        self.multiple_walkers = multiple_walkers

    def update(self, coordinates: Tensor, pbc_box: Tensor = None):
        """
        Update parameter of bias potential.

        Args:
            coordinate (Tensor):              Tensor of shape (B, A, D). Data type is float.
                                              Position coordinate of atoms in system.
            pbc_box (Tensor, optional):       Tensor of shape (B, D) or (1, D). Data type is float.
                                              Box of periodic boundary condition. Default: None.
        """
        #pylint: disable = unused-argument
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""
        Calculate bias potential.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None.
            pbc_box (Tensor, optional):     Tensor of shape (B, D) or (1, D). Data type is float.
                                            Box of periodic boundary condition. Default: None.

        Returns:
            potential (Tensor), Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Dimension of the simulation system. Usually is 3.
        """

        raise NotImplementedError
