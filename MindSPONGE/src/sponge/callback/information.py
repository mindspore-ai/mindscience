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
Callback to print the information of MD simulation
"""

import time

from mindspore.train.callback import Callback, RunContext
from mindspore import Tensor

from ..optimizer import Updater


class RunInfo(Callback):
    r"""Callback to print the information of MD simulation

    Args:
        print_freq (int):   Frequency to print out the information

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

        self.start_time = float(0)
        self.compile_start_time = float(0)

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
        self.count = 0
        cb_params = run_context.original_args()
        self.use_pbc = cb_params.pbc_box is not None
        if isinstance(cb_params.optimizer, Updater):
            self.use_updater = True
            self.kinetics = cb_params.kinetics.copy().asnumpy().squeeze()
            self.temperature = cb_params.temperature.copy().asnumpy().squeeze()
            if self.use_pbc:
                self.volume = cb_params.volume.copy().asnumpy().squeeze()
                self.pressure = cb_params.pressure.copy().asnumpy().squeeze()

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
        if self.count == 0:
            self.compile_start_time = time.time()
        if self.count % self.print_freq == 0:
            self.start_time = time.time()
            cb_params = run_context.original_args()
            if isinstance(cb_params.coordinate[0], Tensor):
                self.crd = cb_params.coordinate[0].copy().asnumpy().squeeze()
            else:
                self.crd = cb_params.coordinate[0].squeeze()

            if self.use_updater:
                self.kinetics = cb_params.kinetics.copy().asnumpy().squeeze()
                self.temperature = cb_params.temperature.copy().asnumpy().squeeze()

    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        if self.count == 0:
            print('[MindSPONGE] Compilation Time: %1.2fs' % (time.time() - self.compile_start_time))
        if self.count % self.print_freq == 0:
            cb_params = run_context.original_args()
            step = cb_params.cur_step
            self.potential = cb_params.potential.copy().asnumpy().squeeze()
            if self.use_updater:
                self.tot_energy = self.potential + self.kinetics
            info = 'Step: '+str(step) + ', '
            info += 'E_pot: ' + str(self.potential)
            if self.use_updater:
                info += ', '
                self.tot_energy = self.potential + self.kinetics
                info += 'E_kin: ' + str(self.kinetics) + ', '
                info += 'E_tot: ' + str(self.tot_energy) + ', '
                info += 'Temperature: ' + str(self.temperature)
                if self.use_pbc:
                    info += ', '
                    self.pressure = cb_params.pressure.copy().asnumpy().squeeze()
                    info += 'Pressure: ' + str(self.pressure) + ', '
                    self.volume = cb_params.volume.copy().asnumpy().squeeze()
                    info += 'Volume: ' + str(self.volume)
            if cb_params.analyse is not None:
                metrics = cb_params.analyse()
                for k, v in metrics.items():
                    info += ', '
                    info += k + ': ' + str(v.squeeze())
            info += ', Time: %1.2fms' % ((time.time() - self.start_time) * 1000)
            print('[MindSPONGE]', info)

        self.count += 1

    def end(self, run_context: RunContext):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
