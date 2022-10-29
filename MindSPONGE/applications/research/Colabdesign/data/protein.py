# Copyright 2022 Huawei Technologies Co., Ltd
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
"""get protein"""
from string import ascii_uppercase, ascii_lowercase
import numpy as np
import mindsponge.common.residue_constants as residue_constants

# in shared
alphabet_list = list(ascii_uppercase + ascii_lowercase)

MODRES = {'TPO': 'THR', 'CSO': 'CYS', 'SEP': 'SER', 'M3L': 'LYS',
          'MSE': 'MET', 'MLY': 'LYS', 'FME': 'MET', 'HYP': 'PRO',
          '4BF': 'TYR', 'KCX': 'LYS', 'B3E': 'GLU', 'B3D': 'ASP',
          'HSK': 'HIS', 'SAC': 'SER', 'PCA': 'GLU', 'DAL': 'ALA',
          'CME': 'CYS', 'CSD': 'CYS', 'OCS': 'CYS', 'DPR': 'PRO',
          'B3K': 'LYS', 'ALY': 'LYS', 'YCM': 'CYS', 'MLZ': 'LYS',
          'HY3': 'PRO', 'LLP': 'LYS', 'MGN': 'GLN', 'MHS': 'HIS',
          'DBZ': 'ALA', 'DCY': 'CYS', 'DVA': 'VAL', 'NLE': 'LEU',
          'SMC': 'CYS', 'AGM': 'ARG', 'B3A': 'ALA', 'DAS': 'ASP',
          'DLY': 'LYS', 'DSN': 'SER', 'DTH': 'THR', 'GL3': 'GLY',
          'HZP': 'PRO', 'CSX': 'CYS', 'BAL': 'ALA', 'HIC': 'HIS',
          'TRQ': 'TRP', 'B3Y': 'TYR', 'PHI': 'PHE', 'PTR': 'TYR',
          'TYS': 'TYR', 'IAS': 'ASP', 'GPL': 'LYS', 'KYN': 'TRP',
          'SEC': 'CYS'}


def _np_get_cb(N, CA, C):
    '''compute CB placement from N, CA, C'''
    return _np_extend(C, N, CA, 1.522, 1.927, -2.143)


def _np_norm(x, axis=-1, keepdims=True, eps=1e-8):
    '''compute norm of vector'''
    return np.sqrt(np.square(x).sum(axis, keepdims=keepdims) + eps)


def _np_extend(a, b, c, lengs, atom, d):
    '''
    given coordinates a-b-c,
    c-d (L)ength, b-c-d (A)ngle, and a-b-c-d (D)ihedral
    return 4th coordinate d
    '''
    normalize = lambda x: x / _np_norm(x)
    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    return c + sum([lengs * np.cos(atom) * bc,
                    lengs * np.sin(atom) * np.cos(d) * np.cross(n, bc),
                    lengs * np.sin(atom) * np.sin(d) * -n])


def pdb_to_string(pdb_file):
    "use pbd to get string"
    modres = {**MODRES}
    lines = []
    for line in open(pdb_file, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()
        if line[:6] == "MODRES":
            x = line[12:15]
            y = line[24:27]
            if x not in modres and y in residue_constants.restype_3to1:
                modres[x] = y
        if line[:6] == "HETATM":
            x = line[17:20]
            if x in modres:
                line = "ATOM  " + line[6:17] + modres.get(x) + line[20:]
        if line[:4] == "ATOM":
            lines.append(line)
    return "\n".join(lines)
