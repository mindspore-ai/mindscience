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
Callback to print the information of MD simulation
"""

from mindspore.train.callback import Callback, RunContext

from ..optimizer import Updater


class RunInfo(Callback):
    r"""
    Callback to print the information of MD simulation.

    Args:
        print_freq (int):   Frequency to print out the information. Default: 1.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, print_freq: int = 1):
        super().__init__()

        self.print_freq = print_freq

        self.potential = None
        self.kinetics = None
        self.temperature = None
        self.pressure = None
        self.tot_energy = None
        self.volume = None

        self.use_pbc = False
        self.use_updater = False

        self.crd = None

        self.count = 0

    def __enter__(self):
        """Return the enter target."""
        return self

    def __exit__(self, *err):
        """Release resources here if have any."""

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        self.use_pbc = cb_params.pbc_box is not None
        if isinstance(cb_params.optimizer, Updater):
            self.use_updater = True
            self.kinetics = cb_params.kinetics.asnumpy().squeeze()
            self.temperature = cb_params.temperature.asnumpy().squeeze()
            if self.use_pbc:
                self.volume = cb_params.volume.asnumpy().squeeze()
                self.pressure = cb_params.pressure.asnumpy().squeeze()

    def epoch_begin(self, run_context: RunContext):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_end(self, run_context: RunContext):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_begin(self, run_context: RunContext):
        """
        Called before each step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        if self.count % self.print_freq == 0:
            cb_params = run_context.original_args()
            self.crd = cb_params.coordinate[0].asnumpy().squeeze()
            if self.use_updater:
                self.kinetics = cb_params.kinetics.asnumpy().squeeze()
                self.temperature = cb_params.temperature.asnumpy().squeeze()

    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        if self.count % self.print_freq == 0:
            cb_params = run_context.original_args()
            step = cb_params.cur_step
            self.potential = cb_params.energy.asnumpy().squeeze()
            if self.use_updater:
                self.tot_energy = self.potential + self.kinetics
            info = 'Step: '+str(step) + ', '
            info += 'E_pot: ' + str(self.potential) + ', '
            if self.use_updater:
                self.tot_energy = self.potential + self.kinetics
                info += 'E_kin: ' + str(self.kinetics) + ', '
                info += 'E_tot: ' + str(self.tot_energy) + ', '
                info += 'Temperature: ' + str(self.temperature)
                if self.use_pbc:
                    self.pressure = cb_params.pressure.asnumpy().squeeze()
                    info += ', Pressure: ' + str(self.pressure)
                    self.volume = cb_params.volume.asnumpy().squeeze()
                    info += ', Volume: ' + str(self.volume)
            print(info)

        self.count += 1

    def end(self, run_context: RunContext):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
