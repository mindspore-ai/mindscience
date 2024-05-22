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
Integrator
"""

from typing import Union, List, Tuple

import mindspore as ms
from mindspore import Tensor
from mindspore.nn import CellList

from .. import Controller
from ..thermostat import Thermostat
from ..barostat import Barostat
from ..constraint import Constraint
from ...system import Molecule
from ...function import get_integer, get_arguments

_INTEGRATOR_BY_KEY = dict()


def _integrator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _INTEGRATOR_BY_KEY:
            _INTEGRATOR_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _INTEGRATOR_BY_KEY:
                _INTEGRATOR_BY_KEY[alias] = cls

        return cls

    return alias_reg


class Integrator(Controller):
    r"""
    Base class for thermostat module in MindSPONGE, which is a subclass of
    :class:`sponge.control.Controller`.

    The :class:`sponge.control.Integrator` module is used to control
    the atomic coordinates and velocities during the simulation process.

    Args:
        system (:class:`sponge.system.Molecule`): Simulation system.
        thermostat (:class:`sponge.control.Thermostat`, optional): Thermostat
            for temperature coupling.
            Default: ``None``.
        barostat (:class:`sponge.control.Barostat`, optional): Barostat for pressure coupling.
            Default: ``None``.
        constraint (Union[:class:`sponge.control.Constraint`, List[:class:`sponge.control.Constraint`]], optional):
            Constraint algorithm. Default: ``None``.

    Inputs:
        - **coordinate** (Tensor) - Coordinate. Tensor of shape :math:`(B, A, D)`.
          Data type is float.
          Here :math:`B` is the number of walkers in simulation,
          :math:`A` is the number of atoms and
          :math:`D` is the spatial dimension of the simulation system, which is usually 3.
        - **velocity** (Tensor) - Velocity. Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **force** (Tensor) - Force. Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **energy** (Tensor) - Energy. Tensor of shape :math:`(B, 1)`. Data type is float.
        - **kinetics** (Tensor) - Kinetics. Tensor of shape :math:`(B, D)`. Data type is float.
        - **virial** (Tensor) - Virial. Tensor of shape :math:`(B, D)`. Data type is float.
        - **pbc_box** (Tensor) - Pressure boundary condition box. Tensor of shape :math:`(B, D)`.
          Data type is float.
        - **step** (int) - Simulation step. Default: ``0``.

    Outputs:
        - coordinate, Tensor of shape :math:`(B, A, D)`. Coordinate. Data type is float.
        - velocity, Tensor of shape :math:`(B, A, D)`. Velocity. Data type is float.
        - force, Tensor of shape :math:`(B, A, D)`. Force. Data type is float.
        - energy, Tensor of shape :math:`(B, 1)`. Energy. Data type is float.
        - kinetics, Tensor of shape :math:`(B, D)`. Kinetics. Data type is float.
        - virial, Tensor of shape :math:`(B, D)`. Virial. Data type is float.
        - pbc_box, Tensor of shape :math:`(B, D)`. Periodic boundary condition box.
          Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge import Molecule
        >>> from sponge.control import Integrator
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> controller = Integrator(system)
    """

    def __init__(self,
                 system: Molecule,
                 thermostat: Thermostat = None,
                 barostat: Barostat = None,
                 constraint: Union[Constraint, List[Constraint]] = None,
                 **kwargs
                 ):

        super().__init__(
            system=system,
            control_step=1,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.acc_unit_scale = Tensor(self.units.acceleration_ref, ms.float32)

        self.thermostat: Thermostat = None
        self.barostat: Barostat = None
        self.constraint: Constraint = None

        self.num_constraint_controller = 0

        self.set_constraint(constraint)
        self.set_thermostat(thermostat)
        self.set_barostat(barostat)

    @classmethod
    def get_name(cls, controller: Controller) -> str:
        r"""
        Get name of controller.

        Args:
            controller ( :class:`sponge.control.Controller`): Controller object.
        """
        if controller is None:
            return None
        if isinstance(controller, Controller):
            return controller.cls_name
        if isinstance(controller, (list, CellList)):
            return [control.cls_name for control in controller]
        raise TypeError(f'The type of controller must be Controller or list but got: {type(controller)}')

    def set_time_step(self, dt: float):
        r"""
        Set simulation time step.

        Args:
            dt (float):  Simulation time step.
        """
        self.time_step = Tensor(dt, ms.float32)
        if self.thermostat is not None:
            self.thermostat.set_time_step(dt)
        if self.barostat is not None:
            self.barostat.set_time_step(dt)
        if self.constraint is not None:
            for i in range(self.num_constraint_controller):
                self.constraint[i].set_time_step(dt)
        return self

    def set_degrees_of_freedom(self, dofs: int):
        r"""
        Set degrees of freedom (DOFs).

        Args:
            dofs (int):  Degrees of freedom.
        """
        self.degrees_of_freedom = get_integer(dofs)
        if self.thermostat is not None:
            self.thermostat.set_degrees_of_freedom(dofs)
        if self.barostat is not None:
            self.barostat.set_degrees_of_freedom(dofs)
        if self.constraint is not None:
            for i in range(self.num_constraint_controller):
                self.constraint[i].set_degrees_of_freedom(dofs)
        return self

    def set_thermostat(self, thermostat: Thermostat):
        r"""
        Set thermostat algorithm for integrator.

        Args:
            thermostat (:class:`sponge.control.Thermostat`): Thermostat algorithm.
        """
        if thermostat is None:
            if self.thermostat is not None:
                print('Set the thermostat to "None"')
            self.thermostat = None
            return self

        if self.thermostat is not None:
            print(f'Set the thermostat to "{thermostat.cls_name}" '
                  f'with reference temperature {thermostat.temperature.asnumpy()} K.')

        self.thermostat = thermostat
        self.thermostat.set_degrees_of_freedom(self.degrees_of_freedom)
        self.thermostat.set_time_step(self.time_step)
        return self

    def set_barostat(self, barostat: Barostat):
        r"""
        Set barostat algorithm for integrator.

        Args:
            barostat (:class:`sponge.control.Barostat`): Barostat algorithm.
        """
        if barostat is None:
            if self.barostat is not None:
                print('Set the barostat to "None"')
            self.barostat = None
            return self

        if self.barostat is not None:
            print(f'Set the barostat to "{barostat.cls_name}" '
                  f'with reference pressure {barostat.pressure.asnumpy()} bar.')

        self.barostat = barostat
        self.barostat.set_degrees_of_freedom(self.degrees_of_freedom)
        self.barostat.set_time_step(self.time_step)
        return self

    def set_constraint(self, constraint: Union[Constraint, List[Constraint]], num_constraints: int = 0):
        r"""
        Set constraint algorithm for integrator.

        Args:
            constraint (Union[:class:`sponge.control.Constraint`, List[:class:`sponge.control.Constraint`]]):
                Constraint algorithm.
            num_constraints (int, optional): Number of constraints.
                Default: ``0``.
        """
        self.num_constraints = num_constraints
        if self.constraint is not None:
            for i in range(self.num_constraint_controller):
                self.num_constraints -= self.constraint[i].num_constraints
            old_name = self.get_name(self.constraint)
            new_name = self.get_name(constraint)
            print(f'Change the constraint from "{old_name} to "{new_name}".')

        self.constraint: List[Constraint] = None
        self.num_constraint_controller = 0
        if constraint is not None:
            if isinstance(constraint, Controller):
                self.num_constraint_controller = 1
                constraint = [constraint]
            elif isinstance(constraint, list):
                self.num_constraint_controller = len(constraint)
            else:
                raise ValueError(f'The type of "constraint" must be '
                                 f'Controller or list but got: {type(constraint)}')

            self.constraint = CellList(constraint)
            for i in range(self.num_constraint_controller):
                self.num_constraints += self.constraint[i].num_constraints
                self.constraint[i].set_time_step(self.time_step)

        self.set_degrees_of_freedom(self.sys_dofs - self.num_constraints)

        return self

    def add_constraint(self, constraint: Constraint):
        r"""
        Add constraint algorithm for integrator.

        Args:
            constraint (:class:`sponge.control.Constraint`):  Constraint algorithm.
        """
        if isinstance(constraint, Controller):
            constraint = [constraint]
            num_constraint_controller = 1
        elif isinstance(constraint, list):
            num_constraint_controller = len(constraint)
        else:
            raise ValueError(f'The type of "constraint" must be '
                             f'Controller or list but got: {type(constraint)}')

        if self.constraint is None:
            return self.set_constraint(constraint)

        self.num_constraint_controller += num_constraint_controller
        self.constraint.extend(constraint)
        for i in range(self.num_constraint_controller):
            self.num_constraints += self.constraint[i].num_constraints
            self.constraint[i].set_time_step(self.time_step)
        degrees_of_freedom = self.sys_dofs - self.num_constraints
        self.set_degrees_of_freedom(degrees_of_freedom)

        return self

    def construct(self,
                  coordinate: Tensor,
                  velocity: Tensor,
                  force: Tensor,
                  energy: Tensor,
                  kinetics: Tensor,
                  virial: Tensor = None,
                  pbc_box: Tensor = None,
                  step: int = 0,
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Update simulation step.

        Args:
            coordinate (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            velocity (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            force (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            energy (Tensor): Tensor of shape :math:`(B, 1)`. Data type is float.
            kinetics (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            virial (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            pbc_box (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            step (int): Simulation step. Default: ``0``.

        Returns:
            - **coordinate** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **velocity** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **force** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **energy** (Tensor) - Tensor of shape :math:`(B, 1)`. Data type is float.
            - **kinetics** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.
            - **virial** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.
            - **pbc_box** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.

        Note:
            :math:`B` is the number of walkers in simulation.
            :math:`A` is the number of atoms.
            :math:`D` is the spatial dimension of the simulation system. Usually is 3.
        """

        raise NotImplementedError
