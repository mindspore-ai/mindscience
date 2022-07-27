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
Molecule
"""
import numpy as np
from mindspore import ms_class
from .residue import Residue


@ms_class
class AminoAcid(Residue):
    r"""Residue of amino acid

    Args:

        name (str):             Name of the residue. Default: ''

        template (dict or str): Template of Residue. Default: None

        atom_name (list):       Atom name. Can be ndarray or list of str. Defulat: None

        start_index (int):      The start index of the first atom in this residue.

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        A:  Number of atoms.

        b:  Number of bonds.

    """

    def __init__(self,
                 name: str = '',
                 template: dict = None,
                 atom_name: str = None,
                 start_index: int = 0,
                 ):

        super().__init__(
            atom_name=atom_name,
            start_index=start_index,
            name=(name.replace('HIE', 'HIS') if 'HIE' in name else name),
            template=template,
        )
        self.atom_index = np.where(
            np.array(list(template[self.name]['atom_name'])) == self.atom_name[0][:, None])[-1]
