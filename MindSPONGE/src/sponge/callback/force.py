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
Callback to write H5MD trajectory file
"""

import numpy as np

from mindspore.train.callback import Callback, RunContext


class ForceEarlyStopping(Callback):
    r"""Set early stopping for simluation according to the norm of force

    Args:

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 delta: float = 1e-2,
                 ):

        self.step = 0
        self.delta = delta

        self.early_stop = False

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        #pylint: disable=unused-argument
        self.step = 0

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

    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()
        force = cb_params.force.copy().asnumpy().squeeze()

        force_norm = np.linalg.norm(force)

        if force_norm < self.delta:
            self.early_stop = True
            print(f'Stop running at step {self.step} with force norm {force_norm}')
            run_context.request_stop()

        self.step += 1

    def end(self, run_context: RunContext):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        #pylint: disable=unused-argument
        if not self.early_stop:
            print(f'Stop running at step {self.step}.')
