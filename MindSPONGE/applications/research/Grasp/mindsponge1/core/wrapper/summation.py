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
"""Energy wrapper"""

from mindspore import Tensor

from .wrapper import EnergyWrapper
from .wrapper import _energy_wrapper_register


@_energy_wrapper_register('sum')
class EnergySummation(EnergyWrapper):
    r"""
    A network to sum the potential and bias directly.

    Args:
        num_walker (int):       Number of multiple walker (B). Default: 1
        dim_potential (int):    Dimension of potential energy (U). Default: 1
        dim_bias (int):         Dimension of bias potential (V). Default: 1

    Outputs:
        energy (Tensor), Tensor of shape (B, 1). Data type is float. Total energy.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 num_walker: int = 1,
                 dim_potential: int = 1,
                 dim_bias: int = 1,
                 ):

        super().__init__(
            num_walker=num_walker,
            dim_potential=dim_potential,
            dim_bias=dim_bias,
            )

    def construct(self, potential: Tensor, bias: Tensor = None):
        """merge the potential and bias.

        Args:
            potential (Tensor): Tensor of shape (B, U). Data type is float.
                                Potential energy.
            bias (Tensor):      Tensor of shape (B, V). Data type is float.
                                Bias potential. Default: None

        Return:
            energy (Tensor), Tensor of shape (B, 1). Data type is float. Total energy.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            U:  Dimension of potential energy.
            V:  Dimension of bias potential.
        """

        potential = self.sum_last_dim(potential)
        if bias is None:
            return potential

        bias = self.sum_last_dim(bias)
        return potential + bias
