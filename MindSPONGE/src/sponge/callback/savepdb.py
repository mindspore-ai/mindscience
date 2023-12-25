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
Callback to write pdb file.
"""

import os
import sys
import signal
import numpy as np
from mindspore.train.callback import Callback, RunContext

from ..system import Molecule
from ..system.modelling.pdb_generator import gen_pdb


class SaveLastPdb(Callback):
    r"""Callback to save the system into a pdb file.

    Args:
        system (Molecule): The system to record.
        pdb_name (str):    The file name of pdb.
        save_freq (int):   Frequency to save the information

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, system: Molecule, pdb_name: str = None, save_freq: int = 10):
        super().__init__()

        self.system = system
        self.last_pdb_name = pdb_name
        self.save_freq = save_freq

        self.convert_to_angstram = self.system.units.convert_length_to('A')
        self.count = 0
        self.count_records = 0

        # Detect process kill signal and save the data.
        signal.signal(signal.SIGINT, self.__stop__)
        signal.signal(signal.SIGTERM, self.__stop__)

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
        print(f'\n\033[33mProgram process terminated. {self.count_records} steps saved in pdb file.\033[0m\n')
        self.close()
        sys.exit(0)

    def close(self):
        self.save_to_pdb()
        return self

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.save_to_pdb()

    def epoch_begin(self, run_context: RunContext):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.save_to_pdb()

    def epoch_end(self, run_context: RunContext):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.save_to_pdb()

    def step_begin(self, run_context: RunContext):
        """
        Called before each step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        if self.count % self.save_freq == 0:
            self.save_to_pdb()


    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        if self.count % self.save_freq == 0:
            self.save_to_pdb()
            self.count_records += 1

        self.count += 1

    def end(self, run_context: RunContext):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.save_to_pdb()

    def save_to_pdb(self):
        """ Save the system information into a pdb file.
        """
        last_resname = self.system.residue_name
        for i, name in enumerate(last_resname):
            last_resname[i] = name[-3:]

        if np.isnan(self.system.coordinate.asnumpy().sum()):
            return None

        # Clear the pdb path.
        if os.path.exists(self.last_pdb_name):
            os.remove(self.last_pdb_name)

        gen_pdb(self.system.coordinate.asnumpy() * self.convert_to_angstram,
                self.system.atom_name[0],
                np.take(last_resname, self.system.atom_resid),
                self.system.atom_resid.asnumpy() + 1,
                pdb_name=self.last_pdb_name)
