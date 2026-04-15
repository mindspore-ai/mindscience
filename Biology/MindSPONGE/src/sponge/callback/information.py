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
from datetime import datetime
import numpy as np

from mindspore.train.callback import Callback, RunContext
from mindspore import Tensor
from mindspore import log as logger
try:
    # MindSpore 1.X
    from mindspore._checkparam import Validator
except ImportError:
    # MindSpore 2.X
    from mindspore import _checkparam as Validator

from ..optimizer import Updater


class RunInfo(Callback):
    r"""Callback to print the information of MD simulation

    Args:

        print_freq (int):   Frequency to print out the information

    Supported Platforms:

        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 per_steps: int = 1,
                 per_epoch: int = 0,
                 show_total_time: bool = True,
                 show_single_time: bool = False,
                 check_force: bool = False,
                 **kwargs
                 ):
        super().__init__()

        if 'print_freq' in kwargs:
            logger.info("`print_freq` will be removed in a future release, please use "
                        "`per_steps` or `per_epoch` instead")
            per_steps = kwargs['print_freq']

        self.per_steps = Validator.check_non_negative_int(per_steps)
        self.per_epoch = Validator.check_non_negative_int(per_epoch)

        if self.per_steps > 0 and self.per_epoch > 0:
            raise ValueError("`per_steps` and `per_epoch` cannot both be greater than zero.")

        self.potential = None
        self.kinetics = None
        self.temperature = None
        self.pressure = None
        self.tot_energy = None
        self.volume = None

        self.use_pbc = False
        self.use_updater = False

        self.crd = None

        self.begin_time = datetime.now()
        self.step_begin_time = datetime.now()
        self.epoch_begin_time = datetime.now()
        self.print_time = time.time()

        self.show_total_time = show_total_time
        self.show_single_time = show_single_time
        self.check_force = check_force

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

        self.begin_time = datetime.now()
        self.print_time = time.time()

        if self.show_total_time:
            print('[MindSPONGE] Started simulation at', self.begin_time.strftime('%Y-%m-%d %H:%M:%S'))

        cb_params = run_context.original_args()

        if cb_params.sink_mode and self.per_steps > 0 and self.per_steps % cb_params.cycle_steps != 0:
            raise ValueError(f"[RunInfo] For per-step output in sink mode, the per_steps must be "
                             f"an integer multiple of the cycle steps ({cb_params.cycle_steps}), "
                             f"but got: {self.per_steps}.")

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
        #pylint: disable=unused-argument
        self.epoch_begin_time = datetime.now()

    def epoch_end(self, run_context: RunContext):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        if self.per_epoch > 0:
            cb_params = run_context.original_args()
            if cb_params.cur_epoch % self.per_epoch == 0:
                self.call_end(run_context)

    def step_begin(self, run_context: RunContext):
        """
        Called before each step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        #pylint: disable=unused-argument
        self.step_begin_time = datetime.now()

    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        if self.per_steps > 0:
            cb_params = run_context.original_args()
            if cb_params.cur_step % self.per_steps == 0:
                self.call_end(run_context)

    def end(self, run_context: RunContext):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        #pylint: disable=unused-argument

        end_time = datetime.now()
        if self.show_total_time:
            print('[MindSPONGE] Finished simulation at', end_time.strftime('%Y-%m-%d %H:%M:%S'))
            used_time = end_time - self.begin_time
            d = used_time.days
            s = used_time.seconds
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            if d >= 1:
                print(f'[MindSPONGE] Simulation time: {d:d} days, {h:d} hours, {m:d} minutes and {s:d} seconds.')
            elif h >= 1:
                print(f'[MindSPONGE] Simulation time: {h:d} hours {m:d} minutes {s:d} seconds.')
            elif m >= 1:
                s += used_time.microseconds / 1e6
                print(f'[MindSPONGE] Simulation time: {m:d} minutes {s:1.1f} seconds.')
            else:
                s += used_time.microseconds / 1e6
                print(f'[MindSPONGE] Simulation time: {s:1.2f} seconds.')
            print('-'*80)

    def call_begin(self, run_context: RunContext):
        """
        Called before each epoch/step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def call_end(self, run_context: RunContext):
        """
        Called after each epoch/step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        step = cb_params.cur_step

        if isinstance(cb_params.coordinate[0], Tensor):
            self.crd = cb_params.coordinate[0].copy().asnumpy().squeeze()
        else:
            self.crd = cb_params.coordinate[0].squeeze()

        if self.use_updater:
            self.kinetics = cb_params.kinetics.copy().asnumpy().squeeze()
            self.temperature = cb_params.temperature.copy().asnumpy().squeeze()

        self.potential = cb_params.potential.copy().asnumpy().squeeze()
        if self.use_updater:
            self.tot_energy = self.potential + self.kinetics
        info = 'Step: '+str(step) + ', '
        info += 'E_pot: ' + str(self.potential)

        if self.check_force:
            force = cb_params.force.copy().asnumpy()
            fnorm = np.linalg.norm(np.sum(force, axis=-2), ord=2, axis=-1).squeeze()
            info += ', F_norm: ' + str(fnorm)

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

        end_time = time.time()
        if self.show_single_time:
            used_time = end_time - self.print_time
            info += ', Time: {:.2f}'.format(used_time) + 's'

        self.print_time = end_time

        print('[MindSPONGE]', info)
