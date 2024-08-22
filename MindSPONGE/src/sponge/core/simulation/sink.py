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
RunOneStepCell
"""

from typing import Tuple
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import functional as F
# from mindspore.common import lazy_inline


try:
    # MindSpore 1.X
    from mindspore._checkparam import Validator
except ImportError:
    # MindSpore 2.X
    from mindspore import _checkparam as Validator

from .run import RunOneStepCell


class SinkCell(Cell):
    r"""Cell for wrap the RunOneStepCell for sink mode

    Args:

    Inputs:

    Outputs:

    Supported Platforms:
        ``Ascend``

    Symbols:
        B:  Batchsize, i.e. number of walkers of the simulation.
        A:  Number of the atoms in the simulation system.
        D:  Spatial dimension of the simulation system. Usually is 3.
    """
    def __init__(self, one_step: RunOneStepCell, cycles: int = 1):
        super().__init__(auto_prefix=False)

        self.one_step = one_step

        self.cycles = Validator.check_non_negative_int(cycles)

    def set_cycles(self, cycles: int):
        r"""
        set cycle steps for sink mode.

        Args:
            cycles(int):  Cycle steps for sink mode.
        """
        self.cycles = Validator.check_non_negative_int(cycles)
        return self

    def construct(self, *inputs):
        """calculate the total potential energy (potential energy and bias potential) of the simulation system.

        Return:
            energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.
                                Total potential energy.

        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.

        """
        energy = None
        force = None
        for _ in range(self.cycles):
            energy, force = self.one_step(*inputs)
        return energy, force


class WithNeighListSinkCell(Cell):
    r"""Cell for wrap the RunOneStepCell with neighbour list for sink mode

    Args:

    Inputs:

    Outputs:

    Supported Platforms:
        ``Ascend``

    Symbols:
        B:  Batchsize, i.e. number of walkers of the simulation.
        A:  Number of the atoms in the simulation system.
        D:  Spatial dimension of the simulation system. Usually is 3.
    """
    # @lazy_inline
    def __init__(self, one_step: RunOneStepCell):
        super().__init__(auto_prefix=False)

        self.one_step = one_step
        if self.one_step.neighbour_list_pace == 0:
            raise ValueError('[WithNeighListSinkCell] The neighbour list pace cannot be ZERO!')

    def update_neighbour_list(self) -> Tuple[Tensor, Tensor]:
        return self.one_step.update_neighbour_list()

    def construct(self, *inputs):
        """calculate the total potential energy (potential energy and bias potential) of the simulation system.

        Return:
            energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.
                                Total potential energy.

        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.

        """
        energy = None
        force = None
        for _ in range(self.one_step.neighbour_list_pace):
            energy, force = self.one_step(*inputs)
        energy = F.depend(energy, self.update_neighbour_list())
        return energy, force


class HyperSinkCell(Cell):
    r"""Cell for wrap the RunOneStepCell for hyper sink mode

    Args:

    Inputs:

    Outputs:

    Supported Platforms:
        ``Ascend``

    Symbols:
        B:  Batchsize, i.e. number of walkers of the simulation.
        A:  Number of the atoms in the simulation system.
        D:  Spatial dimension of the simulation system. Usually is 3.
    """
    def __init__(self, one_step: RunOneStepCell, cycles: int = 1):
        super().__init__(auto_prefix=False)

        self.sink_network = WithNeighListSinkCell(one_step)

        self.cycles = Validator.check_non_negative_int(cycles)

    def set_cycles(self, cycles: int):
        r"""
        set cycle steps for sink mode.

        Args:
            cycles(int):  Cycle steps for sink mode.
        """
        self.cycles = Validator.check_non_negative_int(cycles)
        return self

    def construct(self, *inputs):
        """calculate the total potential energy (potential energy and bias potential) of the simulation system.

        Return:
            energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.
                                Total potential energy.

        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.

        """
        energy = None
        force = None
        for _ in range(self.cycles):
            energy, force = self.sink_network(*inputs)
        return energy, force
