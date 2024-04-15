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

from typing import Tuple, List
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore.nn import Cell

from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn.optim import Optimizer

from .energy import WithEnergyCell
from .force import WithForceCell
from ...function.functions import get_integer, all_none, get_arguments
from ...optimizer import Updater


class RunOneStepCell(Cell):
    r"""
    Cell to run one step simulation.
    This Cell wraps the `energy` and `force` with the `optimizer`. The backward graph will be created
    in the construct function to update the atomic coordinates of the simulation system.

    Args:
        energy(WithEnergyCell): Cell that wraps the simulation system with
                                the potential energy function.
                                Defatul: None
        force(WithForceCell):   Cell that wraps the simulation system with
                                the atomic force function.
                                Defatul: None
        optimizer(Optimizer):   Optimizer for simulation. Defatul: None
        steps(int):             Steps for JIT. Default: 1
        sens(float):            The scaling number to be filled as the input of backpropagation.
                                Default: 1.0
        kwargs(dict):                 other args

    Inputs:
        - **\*inputs** (Tuple(Tensor)) - Tuple of input tensors of `WithEnergyCell`.

    Outputs:
        - energy, Tensor of shape `(B, 1)`. Data type is float. Total potential energy.
        - force, Tensor of shape `(B, A, D)`. Data type is float. Atomic force.

    Note:
        B:  Batchsize, i.e. number of walkers of the simulation.
        A:  Number of the atoms in the simulation system.
        D:  Spatial dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge import WithEnergyCell, RunOneStepCell, Sponge
        >>> from sponge.callback import RunInfo
        >>> # system is the Molecule object defined by user.
        >>> # energy is the Energy object defined by user.
        >>> # opt is the Optimizer object defined by user.
        >>> sim = WithEnergyCell(system, energy)
        >>> one_step = RunOneStepCell(energy=sim, optimizer=opt)
        >>> md = Sponge(one_step)
        >>> run_info = RunInfo(800)
        >>> md.run(2000, callbacks=[run_info])
        [MindSPONGE] Started simulation at 2023-09-04 17:06:26
        [MindSPONGE] Step: 800, E_pot: -150.88245, E_kin: 69.84598, E_tot: -81.03647, Temperature: 418.42694
        [MindSPONGE] Step: 1600, E_pot: -163.72491, E_kin: 57.850487, E_tot: -105.87443, Temperature: 346.56543
        [MindSPONGE] Finished simulation at 2023-09-04 17:07:13
        [MindSPONGE] Simulation time: 47.41 seconds.
    """
    def __init__(self,
                 energy: WithEnergyCell = None,
                 force: WithForceCell = None,
                 optimizer: Optimizer = None,
                 steps: int = 1,
                 sens: float = 1.0,
                 **kwargs
                 ):

        super().__init__(auto_prefix=False)
        self._kwargs = get_arguments(locals(), kwargs)

        if all_none([energy, force]):
            raise ValueError('energy and force cannot be both None!')

        self._neighbour_list_pace = None

        self.system_with_energy = energy
        if self.system_with_energy is not None:
            self.system = self.system_with_energy.system
            self.units = self.system_with_energy.units
            self.system_with_energy.set_grad()
            self._neighbour_list_pace = self.system_with_energy.neighbour_list_pace

        self.system_with_force = force
        if self.system_with_force is not None:
            self.system_with_force.set_grad()

            force_pace = self.system_with_force.neighbour_list_pace

            if self.system_with_energy is None or self._neighbour_list_pace == 0:
                self._neighbour_list_pace = force_pace
            else:
                if force_pace not in (self._neighbour_list_pace, 0):
                    raise ValueError(
                        f'The neighbour_list_pace in WithForceCell ({force_pace}) cannot match '
                        f'the neighbour_list_pace in WithEnergyCell ({self._neighbour_list_pace}).')

        if self.system_with_energy is None:
            self.system = self.system_with_force.system

        self.optimizer = optimizer
        if self.optimizer is None:
            print('[WARNING] No optimizer! The simulation system will not be updated!')

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

    @property
    def neighbour_list_pace(self) -> int:
        r"""
        update step for neighbour list.

        Returns:
            int, the number of steps needed for neighbour list updating.
        """
        return self._neighbour_list_pace

    @property
    def energy_cutoff(self) -> Tensor:
        r"""
        cutoff distance for neighbour list in WithEnergyCell.

        Returns:
            Tensor, cutoff distance for neighbour list in WithEnergyCell.
        """
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.cutoff

    @property
    def force_cutoff(self) -> Tensor:
        r"""
        cutoff distance for neighbour list in WithForceCell.

        Returns:
            Tensor, cutoff distance for neighbour list in WithForceCell.
        """
        if self.system_with_force is None:
            return None
        return self.system_with_force.cutoff

    @property
    def length_unit(self) -> str:
        r"""
        length unit.

        Returns:
            str, length unit.
        """
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        r"""
        energy unit.

        Returns:
            str, energy unit.
        """
        return self.units.energy_unit

    @property
    def num_energies(self) -> int:
        r"""
        number of energy terms :math:`U`.

        Returns:
            int, number of energy terms.
        """
        if self.system_with_energy is None:
            return 0
        return self.system_with_energy.num_energies

    @property
    def energy_names(self):
        r"""
        names of energy terms.

        Returns:
            list[str], names of energy terms.
        """
        if self.system_with_energy is None:
            return []
        return self.system_with_energy.energy_names

    @property
    def bias_names(self) -> List[str]:
        r"""
        name of bias potential energies.

        Returns:
            list[str], the bias potential energies.
        """
        if self.system_with_energy is None:
            return []
        return self.system_with_energy.bias_names

    @property
    def num_biases(self) -> int:
        r"""
        number of bias potential energies :math:`V`.

        Returns:
            int, number of bias potential energies.
        """
        if self.system_with_energy is None:
            return 0
        return self.system_with_energy.num_biases

    @property
    def energies(self) -> Tensor:
        r"""
        Tensor of potential energy components.

        Returns:
            Tensor, Tensor of shape `(B, U)`. Data type is float.
        """
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.energies

    @property
    def biases(self) -> Tensor:
        r"""
        Tensor of bias potential components.

        Returns:
            Tensor, Tensor of shape `(B, V)`. Data type is float.
        """
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.biases

    @property
    def bias(self) -> Tensor:
        r"""
        Tensor of the total bias potential.

        Returns:
            Tensor, Tensor of shape `(B, 1)`. Data type is float.
        """
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.bias

    @property
    def bias_function(self) -> Cell:
        r"""
        Cell of bias potential function.

        Returns:
            Cell, bias potential function.
        """
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.bias_function

    def update_neighbour_list(self):
        r"""update neighbour list."""
        if self.system_with_energy is not None:
            self.system_with_energy.update_neighbour_list()
        if self.system_with_force is not None and self.system_with_force.neighbour_list is not None:
            self.system_with_force.update_neighbour_list()
        return self

    def update_bias(self, step: int):
        r"""
        update bias potential.

        Args:
            step(int):  Simulation step to update bias potential.
        """
        if self.system_with_energy is not None:
            self.system_with_energy.update_bias(step)
        return self

    def update_wrapper(self, step: int):
        r"""
        update energy wrapper.

        Args:
            step(int):  Simulation step to update energy wrapper.
        """
        if self.system_with_energy is not None:
            self.system_with_energy.update_wrapper(step)
        return self

    def update_modifier(self, step: int):
        r"""
        update force modifier.

        Args:
            step(int):  Simulation step to update force modifier.
        """
        if self.system_with_force is not None:
            self.system_with_force.update_modifier(step)
        return self

    def set_pbc_grad(self, value: bool):
        r"""
        set whether to calculate the gradient of PBC box.

        Args:
            value(bool): Flag to judge whether to calculate the gradient of PBC box.
        """
        if self.system_with_energy is not None:
            self.system_with_energy.set_pbc_grad(value)
        if self.system_with_force is not None:
            self.system_with_force.set_pbc_grad(value)
        return self

    def set_steps(self, steps: int):
        r"""
        set steps for JIT.

        Args:
            steps(int): Simulation step for JIT.
        """
        self.steps = get_integer(steps)
        return self

    def construct(self, *inputs) -> Tuple[Tensor, Tensor]:
        r"""
        Run simulation.

        Args:
            *inputs(list): Inputs of the 'WithEnergyCell'.

        Returns:
            - energy, Tensor of shape `(B, 1)`. Data type is float. Total potential energy.
            - force, Tensor of shape `(B, A, D)`. Data type is float. Atomic force.

        Note:
            B:  Batchsize, i.e. number of walkers of the simulation.
            A:  Number of the atoms in the simulation system.
            D:  Spatial dimension of the simulation system. Usually is 3.
        """

        def _run_one_step(*inputs):
            r"""
            Run one step simulation.

            Args:
                *inputs(Tuple(Tensor)): Tuple of input tensors of `WithEnergyCell`.

            Returns:
                - energy, Tensor of shape `(B, 1)`. Data type is float. Total potential energy.
                - force, Tensor of shape `(B, A, D)`. Data type is float. Atomic force.

            Note:
                B:  Batchsize, i.e. number of walkers of the simulation.
                A:  Number of the atoms in the simulation system.
                D:  Spatial dimension of the simulation system. Usually is 3.
            """
            energy = 0
            force = 0
            virial = None
            if self.system_with_energy is not None:
                energy = self.system_with_energy(*inputs)

                sens = F.fill(energy.dtype, energy.shape, self.sens)
                grads = self.grad(self.system_with_energy, self.weights)(*inputs, sens)

                force = -grads[0]
                if len(grads) > 1:
                    virial = 0.5 * grads[1] * self.system.pbc_box

            if self.system_with_force is not None:
                energy, force, virial = self.system_with_force(energy, force, virial)

            if self.optimizer is not None:
                if self.use_updater:
                    energy = F.depend(energy, self.optimizer(energy, force, virial))
                else:
                    grads = (-force,)
                    energy = F.depend(energy, self.optimizer(grads))

            return energy, force

        if self.steps == 1:
            return _run_one_step(*inputs)

        energy = None
        force = None
        for _ in range(self.steps):
            energy, force = _run_one_step(*inputs)

        return energy, force
