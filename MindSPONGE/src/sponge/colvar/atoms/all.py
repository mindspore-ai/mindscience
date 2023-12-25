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
Center of atoms
"""

from mindspore import Tensor
from mindspore.ops import functional as F

from .atoms import AtomsBase
from ...function import all_none, all_not_none, get_integer


class AllAtoms(AtomsBase):
    r"""All atoms of the simulation system.

    Args:
        system (Molecule):  Simulation system. Default: ``None``.

        num_atoms (int):    Number of atoms. The number of atoms must be given when `system` is None.
                            Default: ``None``.

        keep_in_box (bool): Whether to displace the coordinate in PBC box.
                            Default: ``False``.

        dimension (int):    Spatial dimension of the simulation system. Default: 3

        name (str):         Name of the Colvar. Default: 'all_atoms'

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 system=None,
                 num_atoms: int = None,
                 keep_in_box: bool = False,
                 dimension: int = 3,
                 name: str = 'all_atoms',
                 ):

        super().__init__(
            keep_in_box=keep_in_box,
            name=name,
        )

        if all_none([system, num_atoms]):
            raise ValueError('No input!')

        if all_not_none([system, num_atoms]):
            raise ValueError('system and num_atoms cannot be both None!')

        if num_atoms is not None:
            self._set_shape((num_atoms, get_integer(dimension)))
        if system is not None:
            self._set_shape(system.shape)

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""calculate the position of the center of specific atom(s)

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system.
                                    B means Batchsize, i.e. number of walkers in simulation.
                                    A means Number of colvar in system.
                                    D means Dimension of the simulation system. Usually is 3.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            center (Tensor):        Tensor of shape (B, ..., D). Data type is float.
                                    Position of the center of the atoms.
        """

        if self.do_reshape:
            new_shape = coordinate.shape[0] + self._shape
            coordinate = F.reshape(coordinate, new_shape)
        if self.keep_in_box:
            coordinate = self.coordinate_in_pbc(coordinate, pbc_box)

        # (B, ..., D) or (B, ..., 1, D)
        return coordinate
