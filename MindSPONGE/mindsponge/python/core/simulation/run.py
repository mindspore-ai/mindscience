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
"""
RunOneStepCell
"""

from mindspore import ops
from mindspore.ops import functional as F
from mindspore import jit
from mindspore.nn import Cell

from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn.optim import Optimizer

from .simulation import SimulationCell
from ...function.functions import get_integer
from ...optimizer import Updater


class RunOneStepCell(Cell):
    r"""
    Core cell to run one step simulation.

    Args:
        network (SimulationCell):   Network for simulation system.
        optimizer (Optimizer):      Optimizer for simulation.
        steps (int):                Steps for JIT. Default: 1
        sens (float):               The scaling number to be filled as the input of backpropagation.
                                    Default: 1.0

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 network: SimulationCell,
                 optimizer: Optimizer,
                 steps: int = 1,
                 sens: float = 1.0,
                 ):

        super().__init__(auto_prefix=False)

        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.neighbour_list = self.network.neighbour_list
        self.update_neighbour_list = self.network.update_neighbour_list

        self.coordinate = self.network.coordinate
        self.pbc_box = self.network.pbc_box

        self.use_updater = isinstance(self.optimizer, Updater)
        self.weights = self.optimizer.parameters

        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (
            ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(
                self.weights, self.mean, self.degree)

        self.steps = get_integer(steps)

    def set_pbc_grad(self, value: bool):
        """
        set whether to calculate the gradient of PBC box.

        Args:
            value (bool):   Use to judge whether to calculate the gradient of PBC box.
        """
        self.network.set_pbc_grad(value)
        return self

    def set_steps(self, steps: int):
        """
        set steps for JIT.

        Args:
            steps (int):    steps of JIT.
        """
        self.steps = get_integer(steps)
        return self

    @jit
    def get_energy_and_force(self, *inputs):
        """
        get energy and force of the system.

        Returns:
            - energy (Tensor).
            - force (Tensor).
        """
        energy = self.network(*inputs)
        sens = F.fill(energy.dtype, energy.shape, self.sens)
        force = - self.grad(self.network, self.coordinate)(*inputs, sens)
        return energy, force

    # @jit
    def run_one_step(self, *inputs):
        """
        run one step simulation.

        Returns:
            - energy (Tensor), the result of simulation cell.
            - force (Tensor), the result of simulation cell.
        """
        energy = self.network(*inputs)

        sens = F.fill(energy.dtype, energy.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)

        force = -grads[0]

        if self.use_updater:
            energy = F.depend(energy, self.optimizer(grads, energy))
        else:
            energy = F.depend(energy, self.optimizer(grads))

        return energy, force

    def construct(self, *inputs):
        """
        run simulation

        Returns:
            - energy (Tensor), the result of simulation cell.
            - force (Tensor), the result of simulation cell.
        """
        if self.steps == 1:
            return self.run_one_step(*inputs)

        energy = None
        force = None
        for _ in range(self.steps):
            energy, force = self.run_one_step(*inputs)

        return energy, force
