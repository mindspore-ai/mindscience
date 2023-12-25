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
"""Base cell for bais potential"""

from mindspore import Tensor

from ...potential.energy import EnergyCell
from ...colvar import Colvar, get_colvar
from ...function import get_integer


class Bias(EnergyCell):
    r"""Basic cell for bias potential

    Args:
        name (str):         Name of the bias potential. Default: 'bias'

        colvar (Colvar):    Collective variables. Default: ``None``.

        update_pace (int):  Frequency for updating bias potential. Default: 0

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: ``None``.

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: ``None``.

        use_pbc (bool):     Whether to use periodic boundary condition.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 name: str = 'bias',
                 colvar: Colvar = None,
                 update_pace: int = 0,
                 length_unit: str = None,
                 energy_unit: str = None,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            name=name,
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
        )

        self.update_pace = get_integer(update_pace)
        if self.update_pace < 0:
            raise ValueError(f'update_pace cannot be smaller than 0 but got {self.update_pace}')

        self.colvar = get_colvar(colvar)

    def update(self, coordinate: Tensor, pbc_box: Tensor = None) -> Tensor:
        """update parameter of bias potential"""
        #pylint: disable=unused-argument
        return None

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate bias potential.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: ``None``.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: ``None``.
            neighbour_vector (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Vectors from central atom to neighbouring atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: ``None``.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: ``None``.

        Returns:
            potential (Tensor): Tensor of shape (B, 1). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        #pylint: disable=arguments-differ

        raise NotImplementedError
