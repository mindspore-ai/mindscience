# Copyright 2021-2024 @ Shenzhen Bay Laboratory &
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
Callback to write xyz file.
"""

import numpy as np
from mindspore.train.callback import Callback, RunContext

from ..system import Molecule

class XyzCallback(Callback):
    r"""Callback to save the system into a xyz file.

    Args:
        system (Molecule): The system to record.
        xyz_name (str):    The file name of xyz file.
        save_freq (int):   Frequency to save the information.

    Supported Platforms:

        ``Ascend``

    """
    def __init__(self, system: Molecule, xyz_name: str = None, save_freq: int = 10):
        super().__init__()
        self.system = system
        self.xyz_name = xyz_name
        self.save_freq = save_freq
        self.convert_to_angstram = self.system.units.convert_length_to('A')
        self.count = 0
        self.count_records = 0

    def __enter__(self):
        """Return the enter target."""
        return self

    def __exit__(self, *err):
        """Release resources here if have any."""

    def __stop__(self, signal_, frame_):
        """
        Save data when process killed.
        """
        # pylint: disable=unused-argument
        print(f'\n\033[33mProgram process terminated. {self.count_records} steps saved in xyz file.\033[0m\n')

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        # cb_params = run_context.original_args()
        # step = cb_params.cur_step
        # self.save_to_xyz(step)

    def epoch_begin(self, run_context: RunContext):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        # cb_params = run_context.original_args()
        # step = cb_params.cur_step
        # self.save_to_xyz(step)

    def epoch_end(self, run_context: RunContext):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        # cb_params = run_context.original_args()
        # step = cb_params.cur_step
        # self.save_to_xyz(step)

    def step_begin(self, run_context: RunContext):
        """
        Called before each step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        if self.count % self.save_freq == 0:
            cb_params = run_context.original_args()
            step = cb_params.cur_step
            self.save_to_xyz(step)

    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        _ = run_context

        if self.count % self.save_freq == 0:
            # cb_params = run_context.original_args()
            # step = cb_params.cur_step
            # self.save_to_xyz(step)
            self.count_records += 1

        self.count += 1

    def end(self, run_context: RunContext):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        step = cb_params.cur_step
        self.save_to_xyz(step)

    def save_to_xyz(self, step=None):
        """ Save the system information into a pdb file.
        """
        last_resname = self.system.residue_name
        for i, name in enumerate(last_resname):
            last_resname[i] = name[-3:]

        if np.isnan(self.system.coordinate.sum().asnumpy()):
            raise ValueError('Not A Number detected in coordinate!')

        crd = self.system.coordinate.asnumpy()[0] * self.convert_to_angstram
        with open(self.xyz_name, 'a+') as xyz:
            xyz.write('{}\n'.format(crd.shape[-2]))
            if step is None:
                xyz.write('{}\n'.format('Molecule'))
            else:
                xyz.write('{}\n'.format('Molecule at STEP: {}'.format(step)))
            for i in range(crd.shape[-2]):
                xyz.write('{}\t{}\t{}\t{}\n'.format(self.system.atom_name[0][i],
                                                    crd[i][0],
                                                    crd[i][1],
                                                    crd[i][2]))
