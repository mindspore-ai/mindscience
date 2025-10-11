# Copyright 2024 Huawei Technologies Co., Ltd
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
"""inference utils file"""
import mindspore.ops as ops
import mindspore.numpy as np

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

def count_consecutive_occurrences(lst):
    """
    Return the number of consecutive occurrences of each digit in the list.

    Args:
        lst (list): The input list

    Returns:
        list: List of numbers of consecutive occurrences of each digit in the list.
    """
    if not lst:
        return []

    counts = []
    current_count = 1

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1

    counts.append(current_count)

    return counts

def lattices_to_params_ms(lattices):
    """Batched MindSpore version to compute lattice params from matrix.

    Args:
        lattices (Tensor): Tensor of shape (N, 3, 3)
    Returns:
        lengths (Tensor): Tensor of shape (N, 3), unit A
        angles (Tensor):: Tensor of shape (N, 3), unit degree
    """

    lengths = ops.sqrt(ops.reduce_sum(lattices ** 2, -1))

    angles = ops.zeros_like(lengths)

    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3

        cos_angle = ops.clamp(ops.reduce_sum(lattices[..., j, :] * lattices[..., k, :], -1) /
                              (lengths[..., j] * lengths[..., k]), -1.0, 1.0)

        angles[..., i] = ops.acos(cos_angle) * 180.0 / np.pi

    return lengths, angles
