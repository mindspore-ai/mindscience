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

from typing import Union, List

import mindspore as ms
from mindspore import Tensor
from mindspore.nn import CellList

from .. import Controller
from ..thermostat import Thermostat
from ..barostat import Barostat
from ..constraint import Constraint
from ...system import Molecule
from ...function.functions import get_integer


class Integrator(Controller):
    r"""Base class for thermostat module in MindSPONGE, which is a subclass of `Controller`.

        The `Integrator` module used to control the atomic coordinates and velocities during the simulation process.

    Args:

        system (Molecule):          Simulation system

        thermostat (Thermostat):    Thermostat for temperature coupling. Default: None

        barostat (Barostat):        Barostat for pressure coupling. Default: None

        constraint (Union[Constraint, list]):
                                    Constraint algorithm. Default: None


    Supported Platforms:

        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 system: Molecule,
                 thermostat: Thermostat = None,
                 barostat: Barostat = None,
                 constraint: Union[Constraint, List[Constraint]] = None,
                 ):

        super().__init__(
            system=system,
            control_step=1,
        )

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
        """get name of controller"""
        if controller is None:
            return None
        if isinstance(controller, Controller):
            return controller.cls_name
        if isinstance(controller, (list, CellList)):
            return [control.cls_name for control in controller]
        raise TypeError(f'The type of controller must be Controller or list but got: {type(controller)}')

    def set_time_step(self, dt: float):
        """set simulation time step"""
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
        """set degrees of freedom (DOFs)"""
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
        """set thermostat algorithm for integrator"""
        if self.thermostat is not None:
            print(f'Change the thermostat from "{self.thermostat.cls_name} to "{thermostat.cls_name}".')
        if thermostat is None:
            self.thermostat = None
            return self

        self.thermostat = thermostat
        self.thermostat.set_degrees_of_freedom(self.degrees_of_freedom)
        self.thermostat.set_time_step(self.time_step)
        return self

    def set_barostat(self, barostat: Barostat):
        """set barostat algorithm for integrator"""
        if self.barostat is not None:
            print(f'Change the barostat from "{self.barostat.cls_name} to "{barostat.cls_name}".')
        if barostat is None:
            self.barostat = None
            return self

        self.barostat = barostat
        self.barostat.set_degrees_of_freedom(self.degrees_of_freedom)
        self.barostat.set_time_step(self.time_step)
        return self

    def set_constraint(self, constraint: Union[Constraint, list], num_constraints: int = 0):
        """set constraint algorithm for integrator"""
        self.num_constraints = num_constraints
        if self.constraint is not None:
            for i in range(self.num_constraint_controller):
                self.num_constraints -= self.constraint[i].num_constraints
            old_name = self.get_name(self.constraint)
            new_name = self.get_name(constraint)
            print(f'Change the constraint from "{old_name} to "{new_name}".')

        if constraint is None:
            self.constraint = None
            self.num_constraint_controller = 0
        else:
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
        """add constraint algorithm for integrator"""
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
                  ):
        r"""update simulation step.

        Args:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
            velocity (Tensor):      Tensor of shape `(B, A, D)`. Data type is float.
            force (Tensor):         Tensor of shape `(B, A, D)`. Data type is float.
            energy (Tensor):        Tensor of shape `(B, 1)`. Data type is float.
            kinetics (Tensor):      Tensor of shape `(B, D)`. Data type is float.
            virial (Tensor):        Tensor of shape `(B, D)`. Data type is float.
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
            step (int):             Simulation step. Default: 0

        Returns:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
            velocity (Tensor):      Tensor of shape `(B, A, D)`. Data type is float.
            force (Tensor):         Tensor of shape `(B, A, D)`. Data type is float.
            energy (Tensor):        Tensor of shape `(B, 1)`. Data type is float.
            kinetics (Tensor):      Tensor of shape `(B, D)`. Data type is float.
            virial (Tensor):        Tensor of shape `(B, D)`. Data type is float.
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.

        Symbols:
            B:  Number of walkers in simulation.
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        raise NotImplementedError
