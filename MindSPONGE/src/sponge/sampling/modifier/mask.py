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
"""Force modifier"""

from typing import Union, Tuple
from numpy import ndarray
from mindspore import bool_
from mindspore import Tensor
from mindspore.ops import functional as F

from .modifier import ForceModifier
from ...function import get_ms_array


class MaskedDriven(ForceModifier):
    r"""Only drive part of atoms via modifying atomic force.

    Args:
        mask (Union[Tensor, ndarray]): Array of atomic mask to calculate the force.
            The shape of array is `(A)` or `(B, A)`, and the type is bool.

        update_pace (int): Frequency for updating the modifier. Default: 0

        length_unit (str): Length unit. If None is given, it will be assigned with the global length unit.
            Default: ``None``.

        energy_unit (str): Energy unit. If None is given, it will be assigned with the global energy unit.
            Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 mask: Union[Tensor, ndarray],
                 update_pace: int = 0,
                 length_unit: str = None,
                 energy_unit: str = None,
                 ):
        super().__init__(
            update_pace=update_pace,
            length_unit=length_unit,
            energy_unit=energy_unit
        )

        # (A) or (B, A)
        self.mask = get_ms_array(mask, bool_)

    def construct(self,
                  energy: Tensor = 0,
                  energy_ad: Tensor = 0,
                  force: Tensor = 0,
                  force_ad: Tensor = 0,
                  virial: Tensor = None,
                  virial_ad: Tensor = None,
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        aggregate atomic force.

        Args:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Potential energy from ForceCell.
            energy_ad (Tensor): Tensor of shape (B, 1). Data type is float.
                                Potential energy from EnergyCell.
            force (Tensor):     Tensor of shape (B, A, D). Data type is float.
                                Atomic forces from ForceCell.
            force_ad (Tensor):  Tensor of shape (B, A, D). Data type is float.
                                Atomic forces calculated by automatic differentiation.
            virial (Tensor):    Tensor of shape (B, D). Data type is float.
                                Virial calculated from ForceCell.
            virial_ad (Tensor): Tensor of shape (B, D). Data type is float.
                                Virial calculated calculated by automatic differentiation.

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Totoal potential energy for simulation.
            force (Tensor):     Tensor of shape (B, A, D). Data type is float.
                                Total atomic force for simulation.
            virial (Tensor):    Tensor of shape (B, D). Data type is float.
                                Total virial for simulation.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.
        """

        force = force + force_ad
        energy = energy + energy_ad

        if virial is not None or virial_ad is not None:
            if virial is None:
                virial = 0
            if virial_ad is None:
                virial_ad = 0
            virial = virial + virial_ad

        # (B, A, D) * (A, 1) OR (B, A, D) * (B, A, 1)
        force = force * F.expand_dims(self.mask, -1)
        return energy, force, virial
