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
"""Replica exchange molecular dynamics (REMD) """

from typing import Tuple
from mindspore import Tensor

from .wrapper import EnergyWrapper


class ReplicaExchange(EnergyWrapper):
    r"""TODO: Replica exchange molecular dynamics (REMD).

    Args:
        num_walker (int):       Number of multiple walker (B). Default: 1

        dim_potential (int):    Dimension of potential energy (U). Default: 1

        dim_bias (int):         Dimension of bias potential (V). Default: 1

    """

    def __init__(self,
                 num_walker: int,
                 dim_potential: int,
                 dim_bias: int,
                 ):

        super().__init__()

        self.num_walker = num_walker
        self.dim_potential = dim_potential
        self.dim_bias = dim_bias

        print('[Warning] Wrapper REMD is not yet implemented!')

    def construct(self, potentials: Tensor, biases: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        """merge the potential energies and bias potential energies.

        Args:
            potentials (Tensor):    Tensor of shape (B, U). Data type is float.
                                    Potential energies.
            biases (Tensor):        Tensor of shape (B, V). Data type is float.
                                    Bias potential energies. Default: ``None``.

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Total energy (potential energy and bias energy).
            bias (Tensor):      Tensor of shape (B, 1). Data type is float.
                                Total bias potential used for reweighting calculation.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation.
            U:  Dimension of potential energy.
            V:  Dimension of bias potential.

        """

        raise NotImplementedError
