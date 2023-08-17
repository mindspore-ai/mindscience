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
Export xyz files.
"""

import os
from numpy import ndarray
from ...function.functions import get_ndarray


def export_xyz(filename: str, atom: ndarray, coordinate: ndarray, mol_name: str = '', accuracy: str = '{:>12.6f}'):
    """export xyx file"""
    atom = get_ndarray(atom)
    coordinate = get_ndarray(coordinate)
    natom = atom.shape[-1]
    if coordinate.shape[-2] != natom:
        raise ValueError(f'The penultimate dimension of coordinate ({coordinate.shape[-2]}) must be equal to '
                         f'the number of atoms ({natom})!')
    with open(filename, mode='w+') as ofile:
        ofile.write(str(natom)+os.linesep)
        ofile.write(' '+mol_name+os.linesep)
        for a, r in zip(atom, coordinate):
            ofile.write('{:>3d}'.format(a))
            for ri in r:
                ofile.write(accuracy.format(ri))
            ofile.write(os.linesep)
