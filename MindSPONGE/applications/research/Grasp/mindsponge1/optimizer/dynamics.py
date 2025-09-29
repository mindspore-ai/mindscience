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
"""Space updater"""

from mindspore import Tensor
from mindspore.nn.optim.optimizer import opt_init_args_register

from . import Updater
from ..system import Molecule
from ..control.controller import Controller
from ..control.integrator import Integrator


class DynamicUpdater(Updater):
    r"""
    A updater for molecular dynamics (MD) simulation.

    Args:
        system (Molecule):          Simulation system.
        integrator (Integrator):    MD integrator.
        thermostat (Controller):    Thermostat for temperature coupling. Default: None
        barostat (Controller):      Barostat for pressure coupling. Default: None
        constraint (Controller):    Constraint for bond. Default: None
        controller (Controller):    Other controllers. Default: None
        time_step (float):          Time step. Default: 1e-3
        velocity (Tensor):          Tensor of shape (B, A, D). Data type is float.
                                    Default: None
        weight_decay (float):       A value for the weight decay. Default: 0.0
        loss_scale (float):         A value for the loss scale. Default: 1.0

    Returns:
        bool, update the parameters of system.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        A:  Number of atoms.
        D:  Dimension of the simulation system. Usually is 3.

    """
    @opt_init_args_register
    def __init__(self,
                 system: Molecule,
                 integrator: Integrator,
                 thermostat: Controller = None,
                 barostat: Controller = None,
                 constraint: Controller = None,
                 controller: Controller = None,
                 time_step: float = 1e-3,
                 velocity: Tensor = None,
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

        self.integrator = integrator

        if thermostat is not None:
            self.integrator.set_thermostat(thermostat)
        self.thermostat = self.integrator.thermostat

        if barostat is not None:
            if self.pbc_box is None:
                raise ValueError('Barostat cannot be used for the system without periodic boundary condition.')
            self.integrator.set_barostat(barostat)
        self.barostat = self.integrator.barostat

        if constraint is not None:
            self.integrator.set_constraint(constraint)
        self.constraint = self.integrator.constraint

        self.integrator.set_time_step(self.time_step)
        self.integrator.set_degrees_of_freedom(self.degrees_of_freedom)

    def construct(self, gradients: tuple, loss: Tensor = None):
        """update the parameters of system"""
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)

        coordinate = self.coordinate
        velocity = self.velocity
        force = -gradients[0]
        energy = loss
        kinetics = self.kinetics
        pbc_box = self.pbc_box
        virial = None
        if self.pbc_box is not None:
            virial = self.get_virial(gradients[1], pbc_box)

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
