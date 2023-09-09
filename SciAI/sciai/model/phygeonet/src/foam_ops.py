# Copyright 2023 Huawei Technologies Co., Ltd
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
"""foam operations"""
import re

import numpy as np
from sciai.utils import print_log

pattern_mid_vector = re.compile(r"""
(
\(                                                   # match(
[+\-]?\d+([.]\d*)?([Ee][+-]?\d+)?                    # match figures
(\ )                                                 # match space
[+\-]?\d+([.]\d*)?([Ee][+-]?\d+)?                    # match figures
(\ )                                                 # match space
[+\-]?\d+([.]\d*)?([Ee][+-]?\d+)?                    # match figures
\)                                                   # match )
\n                                                   # match next line
)+                                                   # search greedly
""", re.DOTALL | re.VERBOSE)

pattern_mid_scalar = re.compile(r"""
\(                                                   # match"("
\n                                                   # match next line
(
[+\-]?\d+([.]\d*)?([Ee][+-]?\d+)?                    # match figures
\n                                                   # match next line
)+                                                   # search greedly
\)                                                   # match")"
""", re.DOTALL | re.VERBOSE)


def read_vector_from_file(u_file, load_data_path):
    """
    Arg:
        u_file: The directory path of OpenFOAM vector file (e.g., velocity).
    Return:
        vector: Matrix of vector.
    """
    res_mid = extract_vector(u_file)
    with open(f'{load_data_path}/Utemp.txt', mode='w') as fout:
        fout.write(re.sub(r'[()]', '', res_mid.group()))
    return np.loadtxt(f'{load_data_path}/Utemp.txt')


def read_scalar_from_file(file_name, load_data_path):
    """
    Arg:
        file_name: The file name of OpenFOAM scalar field.
    Return:
        a vector of scalar field.
    """
    res_mid = extract_scalar(file_name)
    with open(f'{load_data_path}/temp.txt', mode='w') as f:
        f.write(re.sub(r'[()]', '', res_mid.group()))
    return np.loadtxt(f'{load_data_path}/temp.txt')


def extract_vector(vector_file):
    """ Function is using regular expression select Vector value out.
    Args:
        vector_file: The directory path of file: vector.
    Returns:
        res_mid: the U as (Ux1,Uy1,Uz1);(Ux2,Uy2,Uz2);........
    """
    with open(vector_file, mode='r') as f:
        line = f.read()
    return pattern_mid_vector.search(line)


def extract_scalar(scalar_file):
    """
    sub function of readTurbStressFromFile Using regular expression to select scalar value out.
    Args:
        scalar_file: The directory path of file of scalar.
    Returns:
        res_mid: scalar selected. you need to use res_mid.group() to see the content.
    """
    with open(scalar_file, mode='r') as f:
        line = f.read()
    return pattern_mid_scalar.search(line)


def convert_of_mesh_to_image_structured_mesh(nx, ny, mesh_file, file_name, load_data_path):
    """convert mesh to image"""
    title = ['x', 'y']
    of_scalar = None
    for fn in file_name:
        if fn[-1] == 'U':
            _ = read_vector_from_file(fn, load_data_path)
            title.append('u')
            title.append('v')
        elif fn[-1] == 'p':
            of_scalar = read_scalar_from_file(fn, load_data_path)
            title.append('p')
        elif fn[-1] == 'T':
            of_scalar = read_scalar_from_file(fn, load_data_path)
            title.append('T')
        elif fn[-1] == 'f':
            of_scalar = read_scalar_from_file(fn, load_data_path)
            title.append('f')
        else:
            print_log('Variable name is not clear')
            raise ValueError('Variable name is not clear')
    n_var = len(title)
    of_mesh = read_vector_from_file(mesh_file, load_data_path)
    ng = of_mesh.shape[0]
    of_case = np.zeros([ng, n_var])
    of_case[:, 0:2] = np.copy(of_mesh[:, 0:2])
    if of_scalar is not None:
        of_case[:, 2] = np.copy(of_scalar)
    of_pic = np.reshape(of_case, (ny, nx, n_var), order='F')
    return of_pic
