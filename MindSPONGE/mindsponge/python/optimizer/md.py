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

from typing import Union, List
from numpy import ndarray

from mindspore import Tensor
from mindspore.nn.optim.optimizer import opt_init_args_register

from . import Updater
from ..system import Molecule
from ..control.controller import Controller
from ..control import Integrator, Thermostat, Barostat, Constraint


class UpdaterMD(Updater):
    r"""A updater for molecular dynamics (MD) simulation, which is the subclass of `Updater`.

        UpdaterMD uses four different `Controller`s to control the different variables in the simulation process.
        The `integrator` is used for update the atomic coordinates and velocities, the `thermostat` is used for
        temperature coupling, the `barostat` is used for pressure coupling, and the `constraint` is used for
        bond constraint.

    Args:

        system (Molecule):          Simulation system.

        integrator (Integrator):    Integrator for MD simulation.

        thermostat (Thermostat):    Thermostat for temperature coupling. Default: None

        barostat (Barostat):        Barostat for pressure coupling. Default: None

        constraint (Union[Constraint, List[Constraint]]):
                                    Constraint controller(s) for bond constraint.

        controller (Union[Controller, List[Controller]]):
                                    Other controller(s). It will work after the four specific controllers above.
                                    Default: None

        time_step (float):          Time step. Defulat: 1e-3

        velocity (Union[Tensor, ndarray, List[float]]):
                                    Array of atomic velocity. The shape of array is `(A, D)` or `(B, A, D)`, and
                                    the data type is float. Default: None

        weight_decay (float):       An value for the weight decay. Default: 0

        loss_scale (float):         A value for the loss scale. Default: 1

    Supported Platforms:

        ``Ascend`` ``GPU``

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        A:  Number of atoms.

        D:  Spatial dimension of the simulation system. Usually is 3.

    """
    @opt_init_args_register
    def __init__(self,
                 system: Molecule,
                 integrator: Integrator,
                 thermostat: Thermostat = None,
                 barostat: Barostat = None,
                 constraint: Union[Constraint, List[Constraint]] = None,
                 controller: Union[Controller, List[Controller]] = None,
                 time_step: float = 1e-3,
                 velocity: Union[Tensor, ndarray, List[float]] = None,
                 weight_decay: float = 0.0,
                 loss_scale: float = 1.0,
                 ):

        super().__init__(
            system=system,
            controller=controller,
            time_step=time_step,
            velocity=velocity,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )

        self.integrator: Integrator = integrator
        self.integrator.set_time_step(self.time_step)

        self.constraint: Constraint = None
        self.set_constraint(constraint)

        self.thermostat: Thermostat = None
        self.set_thermostat(thermostat)

        self.barostat: Barostat = None
        self.set_barostat(barostat)

    def set_thermostat(self, thermostat: Thermostat):
        if thermostat is None:
            self.thermostat = None
        else:
            self.integrator.set_thermostat(thermostat)
            self.thermostat = self.integrator.thermostat
        return self

    def set_barostat(self, barostat: Barostat):
        if barostat is None:
            self.barostat = None
        else:
            if self.pbc_box is None:
                raise ValueError('Barostat cannot be used for the system without periodic boundary condition.')
            self.integrator.set_barostat(barostat)
            self.barostat = self.integrator.barostat
        return self

    def set_constraint(self, constraint: Constraint):
        if constraint is None:
            self.constraint = None
        else:
            self.integrator.set_constraint(constraint, self.num_constraints)
            self.constraint = self.integrator.constraint
            self.set_degrees_of_freedom(self.integrator.degrees_of_freedom)
        return self

    def construct(self, energy: Tensor, force: Tensor, virial: Tensor = None):
        """update the parameters of system"""

        force, virial = self.decay_and_scale_grad(force, virial)

        coordinate = self.coordinate
        velocity = self.velocity
        kinetics = self.kinetics
        pbc_box = self.pbc_box

        step = self.identity(self.step)
        coordinate, velocity, force, energy, kinetics, virial, pbc_box = \
            self.integrator(coordinate, velocity, force, energy, kinetics, virial, pbc_box, step)

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
