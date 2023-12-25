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
"""Potential"""

import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import functional as F

from ..function.functions import get_ms_array
from ..function.operations import GetDistance, GetVector
from ..function.units import Units, GLOBAL_UNITS


class ForceCell(Cell):
    r"""
    Base Cell for calculating atomic forces. It returns three terms: energy, force and virial.

    Note:
        The `energy` cannot be None. If the `energy` cannot be calculated, it needs to be assigned to
        a Tensor of shape `(B,1)` and value 0.
        When under periodic boundary conditions, the `virial` cannot be None. If the `virial` cannot
        be calculated, it needs to be assigned to a Tensor of shape (B,D) and value 0.

    Args:
        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: 'nm'.

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: 'kj/mol'.

        use_pbc (bool):     Whether to use periodic boundary condition.

    returns:
        energy (Tensor), Tensor of shape `(B, 1)`. Data type is float.

        force (Tensor), Tensor of shape `(B, A, D)`. Data type is float.

        virial (Tensor), Tensor of shape `(B, D)`. Data type is float. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation.

        A:  Number of atoms.

        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    def __init__(self,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 use_pbc: bool = None,
                 ):

        super().__init__()

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        if energy_unit is None:
            energy_unit = GLOBAL_UNITS.energy_unit
        self.units = Units(length_unit, energy_unit)

        self._use_pbc = use_pbc

        self._exclude_index: Parameter = None

        self.cutoff = None

        self.get_vector = GetVector(use_pbc)
        self.get_distance = GetDistance(use_pbc=use_pbc)

        self.identity = ops.Identity()

    @property
    def use_pbc(self) -> bool:
        """whether to use periodic boundary condition"""
        return self._use_pbc

    @property
    def length_unit(self) -> str:
        """length unit"""
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        """energy unit"""
        return self.units.energy_unit

    @property
    def exclude_index(self) -> Tensor:
        """exclude index"""
        if self._exclude_index is None:
            return None
        return self.identity(self._exclude_index)

    def set_exclude_index(self, exclude_index: Tensor) -> Tensor:
        """set excluded index"""
        if exclude_index is None:
            self._exclude_index = None
            return self._exclude_index

        exclude_index = get_ms_array(exclude_index, ms.int32)
        if exclude_index.ndim == 2:
            exclude_index = F.expand_dims(exclude_index, 0)
        if exclude_index.ndim != 3:
            raise ValueError(f'The rank of exclude_index must be 2 or 3, '
                             f'but got: {exclude_index.shape}')

        if self._exclude_index is not None and self._exclude_index.shape == exclude_index.shape:
            return F.assign(self._exclude_index, exclude_index)

        # (B,A,Ex)
        self._exclude_index = Parameter(exclude_index, name='exclude_index', requires_grad=False)

        return self._exclude_index

    def set_pbc(self, use_pbc: bool = None):
        """set PBC box"""
        self._use_pbc = use_pbc
        self.get_vector.set_pbc(use_pbc)
        self.get_distance.set_pbc(use_pbc)
        return self

    def set_cutoff(self, cutoff: float, unit: str = None):
        """set cutoff distances"""
        if cutoff is None:
            self.cutoff = None
        else:
            cutoff = get_ms_array(cutoff, ms.float32)
            self.cutoff = self.units.length(cutoff, unit)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate atomic forces.

        Args:
            coordinates (Tensor):           Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: ``None``.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: ``None``.
            neighbour_vector (Tensor):       Tensor of shape (B, A, N, D). Data type is bool.
                                            Vectors from central atom to neighbouring atoms.
            neighbour_distances (Tensor):   Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: ``None``.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: ``None``.

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.
            force (Tensor):     Tensor of shape (B, A, D). Data type is float.
            virial (Tensor):    Tensor of shape (B, D). Data type is float.
                                Default: ``None``.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        #pylint: disable=unused-argument

        # (B, 1)
        energy = F.zeros((coordinate.shape[0], 1), coordinate.dtype)
        # (B, A, D)
        force = F.zeros_like(coordinate)

        virial = None
        if pbc_box is not None:
            # (B, D)
            virial = F.zeros_like(pbc_box)

        return energy, force, virial
