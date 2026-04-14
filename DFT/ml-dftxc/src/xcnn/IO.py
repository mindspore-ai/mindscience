# Copyright 2021 Huawei Technologies Co., Ltd
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
"""IO"""
from __future__ import print_function
import numpy as np


def parse_str(fn):
    """
    str file uses a format similar to .xyz file.

    Line 1      number of atoms
    Line 2      atom    x   y   z   [AA]
    Line 3      atom    x   y   z   [AA]
    ...                             
    Line N      atom    x   y   z   [AA]
    Line N+1    charge  spin    

    The `spin' follows the definition in PySCF, 
    equalling to N(alpha)-N(beta)
    The last line is OPTIONAL.
    If not provided, the default value is 0, 0
    """
    atom_list = list()
    atom_str = ''
    with open(fn) as fp:
        n = int(fp.readline())
        for i in range(n):
            ss = fp.readline().split()
            atom_list.append(ss[0])
            atom_str += "%s %16.8e %16.8e %16.8e; " % (ss[0], float(ss[1]), float(ss[2]), float(ss[3]))

        ss = fp.readline().split()
        if len(ss) > 0:
            charge, spin = int(ss[0]), int(ss[1])
        else:
            charge, spin = 0, 0

    return atom_list, atom_str, charge, spin


def _load_density(fn):
    if fn.endswith('npy'): 
        return np.load(fn)
    else:
        return np.loadtxt(fn)


def load_density(fns):
    return [_load_density[f] for f in fns]

