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
"""Potential"""

import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import functional as F

from ..function.functions import gather_vectors
from ..function.operations import GetDistance, GetVector
from ..function.units import Units, global_units


class PotentialCell(Cell):
    r"""
    Basic cell for potential energy.

    Args:
        cutoff (float):              Cutoff distance. Default: None.
        exclude_index (Tensor):      Tensor of shape (B, A, Ex). Data type is int.
                                     Index of the atoms should be excluded from non-bond interaction.
                                     Default: None.
        length_unit (str):           Length unit for position coordinates. Default: None.
        energy_unit (str):           Energy unit. Default: None.
        units (Units):               Units of length and energy. Default: None.
        use_pbc (bool, optional):    Whether to use periodic boundary condition.
                                     If this is None, that means do not use periodic boundary condition.
                                     Default: None.

    Returns:
        potential (Tensor), Tensor of shape (B, 1). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 cutoff: float = None,
                 exclude_index: Tensor = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 units: Units = None,
                 use_pbc: bool = None,
                 ):

        super().__init__()

        if units is None:
            if length_unit is None and energy_unit is None:
                self.units = global_units
            else:
                self.units = Units(length_unit, energy_unit)
        else:
            if not isinstance(units, Units):
                raise TypeError('The type of units must be "Unit" but get type: '+str(type(units)))
            self.units = units

        self.output_dim = 1

        self.cutoff = None
        if cutoff is not None:
            self.cutoff = Tensor(cutoff, ms.float32)

        self.use_pbc = use_pbc
        self._exclude_index = self._check_exclude_index(exclude_index)

        self.get_vector = GetVector(use_pbc)
        self.get_distance = GetDistance(use_pbc)
        self.gather_atoms = gather_vectors

        self.identity = ops.Identity()

    @property
    def exclude_index(self) -> Tensor:
        """
        exclude index.

        Returns:
            Tensor, exclude index.
        """
        if self._exclude_index is None:
            return None
        return self.identity(self._exclude_index)

    def _check_exclude_index(self, exclude_index: Tensor):
        """check excluded index."""
        if exclude_index is None:
            return None
        exclude_index = Tensor(exclude_index, ms.int32)
        if exclude_index.ndim == 2:
            exclude_index = F.expand_dims(exclude_index, 0)
        if exclude_index.ndim != 3:
            raise ValueError('The rank of exclude_index must be 2 or 3 but got: '
                             + str(exclude_index.shape))
        # (B,A,Ex)
        return Parameter(exclude_index, name='exclude_index', requires_grad=False)

    def set_exclude_index(self, exclude_index: Tensor):
        """
        Set excluded index.

        Args:
            exclude_index (Tensor): Tensor of shape (B, A, Ex). Data type is int.
                                    Index of the atoms should be excluded from non-bond interaction.
                                    Default: None.
        """
        self._exclude_index = self._check_exclude_index(exclude_index)
        return self

    @property
    def length_unit(self):
        return self.units.length_unit

    @property
    def energy_unit(self):
        return self.units.energy_unit

    def set_pbc(self, use_pbc: bool = None):
        """
        Set PBC box.

        Args:
            use_pbc (bool, optional):    Whether to use periodic boundary condition.
                                         If this is None, that means do not use periodic boundary condition.
                                         Default: None.
        """
        self.use_pbc = use_pbc
        self.get_vector.set_pbc(use_pbc)
        self.get_distance.set_pbc(use_pbc)
        return self

    def set_cutoff(self, cutoff: Tensor = None):
        """
        Set cutoff distance.

        Args:
            cutoff (Tensor):         Cutoff distance. Default: None
        """
        self.cutoff = None
        if cutoff is not None:
            self.cutoff = Tensor(cutoff, ms.float32)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate potential energy.

        Args:
            coordinates (Tensor):           Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N, D). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distances (Tensor):   Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            potential (Tensor), Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """
        #pylint: disable=invalid-name

        raise NotImplementedError
