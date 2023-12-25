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

from typing import Union, List
import mindspore as ms
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from .energy import EnergyCell
from ..function.functions import get_integer
from ..function.operations import GetDistance, GetVector


class PotentialCell(EnergyCell):
    r"""
    Base class for potential energy.
    The `PotentialCell` is a special subclass of `EnergyCell`. The main difference with `EnergyCell` is
    that normally `EnergyCell` only outputs one energy term, so that `EnergyCell` returns a Tensor of
    the shape `(B, 1)`. And a `PotentialCell` can output multiple energy items, so it returns a Tensor
    of the shape `(B, E)`. Besides, by default the units of `PotentialCell` are equal to the global units.

    Note:
        B:  Batchsize, i.e. number of walkers in simulation.
        E:  Number of energy terms.

    Args:
        num_energies(int):                      Number of the outputs of energy terms. Default: ``1``.
        energy_names(Union[str, List[str]]):    Names of energy terms. Default: ``"potential"``.
        length_unit(str):                       Length unit. If None is given, it will be assigned
                                                with the global length unit. Default: ``None``.
        energy_unit(str):                       Energy unit. If None is given, it will be assigned
                                                with the global energy unit. Default: ``None``.
        use_pbc(bool):                          Whether to use periodic boundary condition.
        name(str):                              Name of energy. Default: ``"potential"``.
        kwargs(dict):                           Other parameters dictionary.

    Inputs:
        - **coordinates** (Tensor) - Tensor of shape (B, A, D). Data type is float.
          Position coordinate of atoms in system.
        - **neighbour_index** (Tensor) - Tensor of shape (B, A, N). Data type is int.
          Index of neighbour atoms. Default: ``None``.
        - **neighbour_mask** (Tensor) - Tensor of shape (B, A, N). Data type is bool.
          Mask for neighbour atoms. Default: ``None``.
        - **neighbour_vector** (Tensor) - Tensor of shape (B, A, N, D). Data type is bool.
          Vectors from central atom to neighbouring atoms. Default: ``None``.
        - **neighbour_distances** (Tensor) - Tensor of shape (B, A, N). Data type is float.
          Distance between neighbours atoms. Default: ``None``.
        - **pbc_box** (Tensor) - Tensor of shape (B, D). Data type is float. Tensor of PBC box. Default: ``None``.

    Outputs:
        potential, Tensor of shape `(B, E)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 num_energies: int = 1,
                 energy_names: Union[str, List[str]] = 'potential',
                 length_unit: str = None,
                 energy_unit: str = None,
                 use_pbc: bool = None,
                 name: str = 'potential',
                 **kwargs
                 ):

        super().__init__(
            name=name,
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
        )
        self._kwargs = kwargs

        self._num_energies = get_integer(num_energies)
        self._energy_names = []
        if isinstance(energy_names, str):
            self._energy_names = [energy_names] * self._num_energies
        elif isinstance(energy_names, list):
            if len(energy_names) != self._num_energies:
                if len(energy_names) != 1:
                    raise ValueError(f'The number of energy names ({len(energy_names)}) does not match '
                                     f'the number of energ ({self._num_energies})')
                energy_names *= self._num_energies
            self._energy_names = energy_names
        else:
            raise TypeError(f'The type of energy_names must str or list but got "{type(energy_names)}"')

        self._exclude_index = None

        self.get_vector = GetVector(self._use_pbc)
        self.get_distance = GetDistance(use_pbc=self._use_pbc)

    @property
    def exclude_index(self) -> Tensor:
        """
        Exclude index.

        Returns:
            Tensor, exclude index.
        """
        if self._exclude_index is None:
            return None
        return self.identity(self._exclude_index)

    @property
    def num_energies(self) -> int:
        """
        Number of energy components.

        Returns:
            int, number of energy components.
        """
        return self._num_energies

    @property
    def energy_names(self) -> List[str]:
        """
        List of strings of energy names.

        Returns:
            List[str], strings of energy names.
        """
        return self._energy_names

    def set_exclude_index(self, exclude_index: Tensor) -> Tensor:
        """
        Set excluded index.

        Args:
            exclude_index(Tensor):  Excluded index of the system.

        Returns:
            Tensor, excluded index.
        """
        self._exclude_index = self._check_exclude_index(exclude_index)
        return self._exclude_index

    def set_pbc(self, use_pbc: bool = None):
        """
        Set PBC box.

        Args:
            use_pbc(bool):  Whether to use periodic boundary condition.
        """
        self._use_pbc = use_pbc
        self.get_vector.set_pbc(use_pbc)
        self.get_distance.set_pbc(use_pbc)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate potential energy.

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
            potential (Tensor): Tensor of shape (B, E). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.
            E:  Number of energy terms.

        """
        #pylint: disable=unused-argument

        raise NotImplementedError

    def _check_exclude_index(self, exclude_index: Tensor):
        """check excluded index"""
        if exclude_index is None:
            return None
        exclude_index = Tensor(exclude_index, ms.int32)
        if exclude_index.ndim == 2:
            exclude_index = F.expand_dims(exclude_index, 0)
        if exclude_index.ndim != 3:
            raise ValueError(f'The rank of exclude_index must be 2 or 3, '
                             f'but got: {exclude_index.shape}')
        # (B,A,Ex)
        return Parameter(exclude_index, name='exclude_index', requires_grad=False)
