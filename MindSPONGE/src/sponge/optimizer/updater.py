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
"""Space updater"""

from typing import Union, List, Tuple
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import CellList
from mindspore.nn import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register
from mindspore.common.initializer import initializer

from ..system import Molecule
from ..control import Controller
from ..function import get_ms_array, get_arguments
from ..function import functions as func


class Updater(Optimizer):
    r"""
    Base class of the MindSPONGE updater, which is a special subclass of the
    `mindspore.nn.Optimizer` in MindSpore.

    The :class:`sponge.optimizer.Updater` updates the atomic coordinates of
    the simulation system. The updating of atomic coordinates requires atomic
    forces and atomic velocities, where the force is passed from outside and
    the velocity is the parameter of the :class:`sponge.optimizer.Updater`
    itself. And in the case of periodic boundary conditions (PBC), the
    :class:`sponge.optimizer.Updater` could also update the size of the PBC
    box by the virial of the simulation system.

    The :class:`sponge.optimizer.Updater` controls the values of seven
    variables during the simulation through a series of
    :class:`sponge.control.Controller` : coordinates, velocity, force, energy, kinetics,
    virial and pbc_box. If more than one :class:`sponge.control.Controller`
    is passed in, they will work in sequence.

    Args:
        system(:class:`sponge.system.Molecule`): Simulation system.
        controller(Union[:class:`sponge.control.Controller`, List[:class:`sponge.control.Controller`]], optional):
          Controller or list of controllers to control the seven variables
          (coordinate, velocity, force, energy, kinetics, virial and
          pbc_box) of the simulation system. Default: ``None``.
        time_step(float, optional): Time step. Default: ``1e-3``.
        velocity(Union[Tensor, ndarray, List[float]], optional): Array of atomic velocity.
          The shape of array is :math:`(A, D)` or :math:`(B, A, D)`.
          Here :math:`A` is the number of atoms,
          :math:`D` is the spatial dimension of the simulation system,
          which is usually 3.
          The data type is float. Default: ``None``.
        weight_decay(float, optional): An value for the weight decay. Default: ``0.0``.
        loss_scale(float, optional): A value for the loss scale. Default: ``1.0``.
        kwargs(dict): Other arguments.

    Inputs:
        - **energy** (Tensor) - Energy of the system.
          Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **force** (Tensor) - Force of the system.
          Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **virial** (Tensor) - Virial of the system.
          Tensor of shape `(B, A, D)`. Data type is float.
          Default: ``None``.

    Outputs:
        - **success** (bool) - whether successfully finish the current optimization step and move to next step.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge.system import Molecule
        >>> from sponge.optimizer import Updater
        >>> from sponge.potential.forcefield import ForceField
        >>> from sponge.core.sponge import Sponge
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> potential = ForceField(system, parameters='SPCE')
        >>> updater = Updater(system, controller=None, time_step=1e-3)
        >>> print(system.coordinate.value())
        >>> # [[[ 0. 0. 0. ]
        >>> # [ 0.07907964 0.06120793 0. ]
        >>> # [-0.07907964 0.06120793 0. ]]]
        >>> md = Sponge(system, potential, updater)
        >>> # As controller is None, no change on MD variables happens
        >>> md.run(1000)
        >>> # [MindSPONGE] Started simulation at 2024-04-29 01:01:34
        >>> # [MindSPONGE] Finished simulation at 2024-04-29 01:01:37
        >>> # [MindSPONGE] Simulation time: 2.54 seconds.
        >>> print(system.coordinate.value())
        >>> # [[[ 0. 0. 0. ]
        >>> # [ 0.07907964 0.06120793 0. ]
        >>> # [-0.07907964 0.06120793 0. ]]]
    """
    @opt_init_args_register
    def __init__(self,
                 system: Molecule,
                 controller: Union[Controller, List[Controller]] = None,
                 time_step: float = 1e-3,
                 velocity: Union[Tensor, ndarray, List[float]] = None,
                 weight_decay: float = 0.0,
                 loss_scale: float = 1.0,
                 **kwargs
                 ):

        super().__init__(
            learning_rate=time_step,
            parameters=system.space_parameters(),
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )
        self._kwargs = get_arguments(locals(), kwargs)
        self._kwargs.pop('velocity')

        self.time_step = Tensor(time_step, ms.float32)

        self.system = system
        self.coordinate = self.system.coordinate
        self.pbc_box = self.system.pbc_box

        # (B,A)
        self.atom_mass = self.system.atom_mass
        self.inv_mass = self.system.inv_mass
        # (B,A,1)
        self._atom_mass = F.expand_dims(self.atom_mass, -1)
        self._inv_mass = F.expand_dims(self.inv_mass, -1)

        self.num_walker = system.num_walker
        self.num_atoms = system.num_atoms
        self.dimension = system.dimension

        self.units = self.system.units
        self.kinetic_unit_scale = Tensor(self.units.kinetic_ref, ms.float32)

        if velocity is None:
            self.velocity = Parameter(msnp.zeros_like(self.coordinate), name='velocity')
        else:
            velocity = get_ms_array(velocity, ms.float32)
            if velocity.ndim == 2:
                velocity = F.expand_dims(velocity, 0)
            if velocity.shape != self.coordinate.shape:
                raise ValueError(f'The shape of velocity {velocity.shape} must be equal to '
                                 f'the shape of coordinate {self.coordinate.shape}!')
            self.velocity = Parameter(velocity, name='velocity')

        self.num_constraints = 0
        self.num_controller = 0
        self.controller: List[Controller] = None
        if controller is not None:
            if isinstance(controller, Controller):
                self.num_controller = 1
                controller = [controller]
            elif isinstance(controller, list):
                self.num_controller = len(controller)
            else:
                raise TypeError(f'The type of "controller" must be Controller or list but got: {type(controller)}')

            self.controller = CellList(controller)
            for i in range(self.num_controller):
                self.num_constraints += self.controller[i].num_constraints

        self.sys_dofs = system.degrees_of_freedom
        self.degrees_of_freedom = 0
        self.set_degrees_of_freedom(self.sys_dofs - self.num_constraints)

        self.identity = ops.Identity()

        self.kinetics = None
        self.temperature = None
        if self.velocity is not None:
            kinetics = self.get_kinetics(self.velocity)
            temperature = self.get_temperature(kinetics)
            # (B,D)
            self.kinetics = Parameter(kinetics, name="kinetics")
            # (B)
            self.temperature = Parameter(temperature, name="temperature")

        self.virial = None
        self.pressure = None
        if self.pbc_box is not None:
            # (B,D)
            self.virial = Parameter(initializer(
                'zeros', (self.num_walker, self.dimension), ms.float32), name="virial")
            self.pressure = Parameter(initializer(
                'zeros', (self.num_walker, self.dimension), ms.float32), name="pressure")

        self.step = Parameter(Tensor(0, ms.int32), name='updater_step')

        if controller is not None:
            for i in range(self.num_controller):
                self.controller[i].set_time_step(self.time_step)

    @property
    def boltzmann(self) -> float:
        r"""
        Boltzmann constant in current unit.

        Returns:
            float, Boltzmann constant in current unit.
        """
        return self.units.boltzmann

    @property
    def press_unit_scale(self) -> float:
        r"""
        Reference value of pressure.

        Returns:
            float, reference value of pressure.
        """
        return self.units.pressure_ref

    def set_step(self, step: int = 0):
        r"""
        Set current step of the system.

        Args:
            step(int): Current step of the system. Default: 0
        """
        step = Tensor(step, ms.int32)
        F.depend(True, F.assign(self.step, step))
        return self

    def set_degrees_of_freedom(self, dofs: int):
        r"""
        Set degrees of freedom (DOFs)

        Args:
            dofs(int): Degrees of freedom.
        """
        self.degrees_of_freedom = func.get_integer(dofs)
        self.num_constraints = self.sys_dofs - self.degrees_of_freedom
        for i in range(self.num_controller):
            self.controller[i].set_degrees_of_freedom(self.degrees_of_freedom)
        return self

    def update_coordinate(self, coordinate: Tensor, success: bool = True) -> bool:
        r"""
        Update the parameters of coordinate

        Args:
            coordinate(Tensor): Tensor of atomic coordinates. Data type is float.
            success(bool): Whether to update the coordinate. Default: ``True``.

        Returns:
            bool, whether successfully update the coordinate.
        """
        return F.depend(success, F.assign(self.coordinate, coordinate))

    def update_pbc_box(self, pbc_box: Tensor, success: bool = True) -> bool:
        r"""
        Update the parameters of PBC box.

        Args:
            pbc_box(Tensor): Tensor of PBC box. Data type is float.
            success(bool): Whether to update the pbc_box. Default: ``True``.

        Returns:
            bool, whether successfully update the parameters of PBC box.
        """
        if self.pbc_box is None:
            return success
        return F.depend(success, F.assign(self.pbc_box, pbc_box))

    def update_velocity(self, velocity: Tensor, success: bool = True) -> bool:
        r"""
        Update the parameters of velocity.

        Args:
            velocity(Tensor): Tensor of atomic velocities. Data type is float.
            success(bool): Whether to update the velocities. Default: ``True``.

        Returns:
            bool, whether successfully update the parameters of atomic velocities.
        """
        return F.depend(success, F.assign(self.velocity, velocity))

    def update_kinetics(self, kinetics: Tensor, success: bool = True) -> bool:
        r"""
        Update the parameters of kinetics.

        Args:
            kinetics(Tensor): Tensor of kinetics. Data type is float.
            success(bool): Whether to update the kinetics. Default: ``True``.

        Returns:
            bool, whether successfully update the parameters of kinetics.
        """
        if self.kinetics is None:
            return success
        return F.depend(success, F.assign(self.kinetics, kinetics))

    def update_temperature(self, temperature: Tensor, success: bool = True) -> bool:
        r"""
        Update the parameters of temperature.

        Args:
            temperature(Tensor): Tensor of temperature. Data type is float.
            success(bool): Whether to update the temperature. Default: ``True``.

        Returns:
            bool, whether successfully update the parameters of temperature.
        """
        if self.temperature is None:
            return success
        return F.depend(success, F.assign(self.temperature, temperature))

    def update_virial(self, virial: Tensor, success: bool = True) -> bool:
        r"""
        Update the parameters of virial.

        Args:
            virial(Tensor): Tensor of virial. Data type is float.
            success(bool): Whether to update the virial. Default: ``True``.

        Returns:
            bool, whether successfully update the parameters of virial.
        """
        if self.pbc_box is None:
            return success
        return F.depend(success, F.assign(self.virial, virial))

    def update_pressure(self, pressure: Tensor, success: bool = True) -> bool:
        r"""
        Update the parameters of pressure.

        Args:
            pressure(Tensor): Tensor of pressure. Data type is float.
            success(bool): Whether to update the pressure. Default: ``True``.

        Returns:
            bool, whether successfully update the parameters of pressure.
        """
        if self.pbc_box is None:
            return success
        return F.depend(success, F.assign(self.pressure, pressure))

    def get_velocity(self) -> Tensor:
        r"""
        Get velocity.

        Returns:
            Tensor, atom velocities of the system.
        """
        if self.velocity is None:
            return None
        return self.identity(self.velocity)

    def get_kinetics(self, velocity: Tensor) -> Tensor:
        r"""
        Get kinectics.

        Args:
            velocity(Tensor): Tensor of atom velocities. Data type is float.

        Returns:
            Tensor, the kinectics of the system.
        """
        # (B,A,D)
        kinetics = 0.5 * self._atom_mass * velocity**2
        # (B,D) <- (B,A,D)
        kinetics = F.reduce_sum(kinetics, -2)
        return kinetics * self.kinetic_unit_scale

    def get_temperature(self, kinetics: Tensor = None) -> Tensor:
        r"""
        Get temperature.

        Args:
            kinetics(Tensor): Tensor of kinetics. Data type is float. Default: ``None``.

        Returns:
            Tensor, the temperature of the system.
        """
        # (B) <- (B,D)
        kinetics = F.reduce_sum(kinetics, -1)
        return 2 * kinetics / self.degrees_of_freedom / self.boltzmann

    def get_pressure(self, kinetics: Tensor, virial: Tensor, pbc_box: Tensor) -> Tensor:
        r"""
        Get pressure.

        Args:
            kinetics(Tensor): Tensor of kinetics. Data type is float.
            virial(Tensor): Tensor of virial. Data type is float.
            pbc_box(Tensor): Tensor of pbc_box. Data type is float.

        Returns:
            Tensor, the pressure of the system.
        """
        if self.pbc_box is None:
            return None
        # (B,D) = ((B,D) - (B, D)) / (B,1)
        volume = func.keepdims_prod(pbc_box, -1)
        pressure = 2 * (kinetics - virial) / volume
        return pressure * self.press_unit_scale

    def get_dt(self):
        r"""
        Get the learning rate of current step.

        Returns:
            float, the learning rate of current step.
        """
        return self.get_lr()

    def next_step(self, success: bool = True) -> bool:
        r"""
        Finish the current optimization step and move to next step.

        Args:
            success(bool): Whether to finish the current optimization step and move to next step. Default: ``True``.

        Returns:
            bool, whether successfully finish the current optimization step and move to next step.
        """
        return F.depend(success, F.assign(self.step, self.step+1))

    def decay_and_scale_grad(self, force: Tensor, virial: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""
        Do weight decay and gradient scale for force and virial.

        Args:
            force(Tensor): Tensor of force. Data type is float.
            virial(Tensor, optional): Tensor of virial. Data type is float. Default: ``None``.

        Returns:
            - Tensor, Tensor of force after weight decay and gradient scale.
            - Tensor, Tensor of virial after weight decay and gradient scale.
              If pbc_box is None, the output virial is the same as input.
        """
        if self.exec_weight_decay or self.need_scale:
            if self.pbc_box is None:
                gradients = (force,)
            else:
                gradients = (force, virial)

            gradients = self.decay_weight(gradients)
            gradients = self.scale_grad(gradients)

            force = gradients[0]
            if self.pbc_box is not None:
                virial = gradients[1]

        return force, virial

    def construct(self, energy: Tensor, force: Tensor, virial: Tensor = None):
        r"""
        Update the parameters of system

        Args:
            energy (Tensor): Tensor of energy. Data type is float.
            force (Tensor): Tensor of force. Data type is float.
            virial (Tensor): Tensor of virial. Data type is float. Default: ``None``.

        Returns:
            bool, whether successfully finish the current optimization step and move to next step.
        """

        force, virial = self.decay_and_scale_grad(force, virial)

        coordinate = self.coordinate
        velocity = self.velocity
        kinetics = self.kinetics
        pbc_box = self.pbc_box

        step = self.identity(self.step)
        if self.controller is not None:
            for i in range(self.num_controller):
                coordinate, velocity, force, energy, kinetics, virial, pbc_box = \
                    self.controller[i](coordinate, velocity, force, energy, kinetics, virial, pbc_box, step)

        temperature = self.get_temperature(kinetics)
        pressure = self.get_pressure(kinetics, virial, pbc_box)

        success = True
        success = self.update_coordinate(coordinate, success)
        success = self.update_velocity(velocity, success)
        success = self.update_pbc_box(pbc_box, success)
        success = self.update_kinetics(kinetics, success)
        success = self.update_temperature(temperature, success)
        success = self.update_virial(virial, success)
        success = self.update_pressure(pressure, success)

        return self.next_step(success)
