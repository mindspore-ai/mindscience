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
"""Energy wrapper"""

from typing import Tuple
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell

from ...function import Units, GLOBAL_UNITS
from ...function import get_integer, keepdims_sum


class EnergyWrapper(Cell):
    r"""A network to process and merge the potential and bias during the simulation.

    Args:
        update_pace (int):  Frequency for updating the wrapper. Default: 0

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: ``None``.

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 update_pace: int = 0,
                 length_unit: str = None,
                 energy_unit: str = None,
                 ):
        super().__init__(auto_prefix=False)

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        if energy_unit is None:
            energy_unit = GLOBAL_UNITS.energy_unit
        self.units = Units(length_unit, energy_unit)

        self._update_pace = get_integer(update_pace)

        self.identity = ops.Identity()

    @property
    def update_pace(self) -> int:
        """frequency for updating the wrapper"""
        return self._update_pace

    def update(self):
        """update energy wrapper"""
        return

    def construct(self, potentials: Tensor, biases: Tensor = None) -> Tuple[Tensor, Tensor]:
        """merge the potential energies and bias potential energies.

        Args:
            potentials (Tensor):    Tensor of shape `(B, U)`. Data type is float.
                                    Potential energies.
            biases (Tensor):        The shape of tensor is `(B, V)`. The data type is float.
                                    Bias potential energies. Default: ``None``.

        Returns:
            energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.
                                Total energy (potential energy and bias energy).
            bias (Tensor):      Tensor of shape `(B, 1)`. Data type is float.
                                Total bias potential used for reweighting calculation.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation.
            U:  Dimension of potential energy.
            V:  Dimension of bias potential.

        """

        potential = keepdims_sum(potentials, -1)

        if biases is None:
            return potential, 0

        bias = keepdims_sum(biases, -1)
        energy = potential + bias

        return energy, bias
