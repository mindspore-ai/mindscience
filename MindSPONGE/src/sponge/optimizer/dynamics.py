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

from mindspore import Tensor

from .md import UpdaterMD
from ..system import Molecule
from ..control.controller import Controller
from ..control.integrator import Integrator


class DynamicUpdater(UpdaterMD):
    r"""
    Updater for molecular dynamics (MD) simulation.

    This updater will be removed a future release. Please use :class:`sponge.optimizer.UpdaterMD` instead.

    Args:
        system (:class:`sponge.system.Molecule`): Simulation system.
        integrator (`sponge.control.Integrator`): MD integrator.
        thermostat (:class:`sponge.control.Controller`, optional): Thermostat
          for temperature coupling.
          Default: ``None``.
        barostat (:class:`sponge.control.Controller`, optional): Barostat for pressure coupling.
          Default: ``None``.
        constraint (:class:`sponge.control.Controller`, optional): Constraint for bond.
        controller (:class:`sponge.control.Controller`, optional): Other controllers.
        time_step (float, optional): Time step. Default: ``1e-3``.
        velocity (Tensor, optional): Atomic velocity. The shape of tensor is :math:`(B, A, D)`.
          Here :math:`B` is the number of walkers in simulation,
          :math:`A` is the number of atoms,
          and :math:`D` is the spatial dimension of the simulation system, which is usually 3.
          Data type is float. Default: ``None``.
        weight_decay (float, optional): An value for the weight decay. Default: ``0.0``.
        loss_scale (float, optional): A value for the loss scale. Default: ``1.0``.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
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
            integrator=integrator,
            thermostat=thermostat,
            barostat=barostat,
            constraint=constraint,
            controller=controller,
            time_step=time_step,
            velocity=velocity,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )

        print('[WARNING] `DynamicUpdater` will be removed a future release. '
              'Please use `UpdaterMD` instead.')
