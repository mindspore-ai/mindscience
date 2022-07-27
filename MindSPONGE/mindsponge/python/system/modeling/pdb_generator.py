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
Module used to generate a pdb file via crd and res names.
"""

import os


def gen_pdb(crd, atom_names, res_names, res_ids, pdb_name='temp.pdb'):
    """Write protein crd information into pdb format files.
    Args:
        crd(numpy.float32): The coordinates of protein atoms.
        atom_names(numpy.str_): The atom names differ from aminos.
        res_names(numpy.str_): The residue names of amino names.
        res_ids(numpy.int32): A unique mask each same residue.
        pdb_name(str): The path to save the pdb file, absolute path is suggested.
    """
    success = 1
    file = os.open(pdb_name, os.O_RDWR | os.O_CREAT)
    pdb = os.fdopen(file, "w")

    pdb.write('MODEL     1\n')
    for i, c in enumerate(crd[0]):
        pdb.write('ATOM'.ljust(6))
        pdb.write('{}'.format(i + 1).rjust(5))
        if len(atom_names[i]) < 4:
            pdb.write('  ')
            pdb.write(atom_names[i].ljust(3))
        else:
            pdb.write(' ')
            pdb.write(atom_names[i].ljust(4))
        pdb.write(res_names[i].rjust(4))
        pdb.write('A'.rjust(2))
        pdb.write('{}'.format(res_ids[i]).rjust(4))
        pdb.write('    ')
        pdb.write('{:.3f}'.format(c[0]).rjust(8))
        pdb.write('{:.3f}'.format(c[1]).rjust(8))
        pdb.write('{:.3f}'.format(c[2]).rjust(8))
        pdb.write('1.0'.rjust(6))
        pdb.write('0.0'.rjust(6))
        pdb.write('{}'.format(atom_names[i][0]).rjust(12))
        pdb.write('\n')
    pdb.write('TER\n')
    pdb.write('ENDMDL\n')
    pdb.write('END\n')

    pdb.close()
    return success
