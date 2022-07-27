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
Add missing atoms module.
"""
import numpy as np


def rotate_by_axis(axis, theta):
    """Rotate an atom by a given axis with angle theta.
    Args:
        axis: The rotate axis.
        theta: The rotate angle.
    Returns:
        The rotate matrix.
    """
    vx, vy, vz = axis[0], axis[1], axis[2]
    return np.array([[vx*vx*(1-np.cos(theta))+np.cos(theta),
                      vx*vy*(1-np.cos(theta))-vz*np.sin(theta),
                      vx*vz*(1-np.cos(theta))+vy*np.sin(theta)],
                     [vx*vy*(1-np.cos(theta))+vz*np.sin(theta),
                      vy*vy*(1-np.cos(theta))+np.cos(theta),
                      vy*vz*(1-np.cos(theta))-vx*np.sin(theta)],
                     [vx*vz*(1-np.cos(theta))-vy*np.sin(theta),
                      vy*vz*(1-np.cos(theta))+vx*np.sin(theta),
                      vz*vz*(1-np.cos(theta))+np.cos(theta)]])


def add_h(crd, atype=None, i=None, j=None, k=None):
    """Add hydrogen once.
    Args:
        crd: The coordinates of all atoms.
        atype: Different types correspond to different addH algorithms.
    Indexes:
        c6: Add one hydrogen at atom i. j and k atoms are connected to atom i.
    """
    if atype is None:
        raise ValueError('The type of AddH should not be None!')

    if atype != 'h2o' and i is None or j is None or k is None:
        raise ValueError('3 atom indexes are need.')

    if atype == 'c6':
        left_arrow = crd[j] - crd[i]
        left_arrow /= np.linalg.norm(left_arrow)
        right_arrow = crd[k] - crd[i]
        right_arrow /= np.linalg.norm(right_arrow)
        h_arrow = -1 * (left_arrow + right_arrow)
        h_arrow /= np.linalg.norm(h_arrow)
        return (h_arrow + crd[i])[None, :]

    if atype == 'dihedral':
        h_arrow = crd[j] - crd[k]
        h_arrow /= np.linalg.norm(h_arrow)
        return (h_arrow + crd[i])[None, :]

    if atype == 'c2h4':
        h_arrow_1 = crd[j] - crd[k]
        h1 = (h_arrow_1/np.linalg.norm(h_arrow_1) + crd[i])[None, :]
        middle_arrow = (crd[i] - crd[j])
        middle_arrow /= np.linalg.norm(middle_arrow)
        middle_arrow *= np.linalg.norm(h_arrow_1)
        h_arrow_2 = -h_arrow_1 + middle_arrow
        h2 = (h_arrow_2/np.linalg.norm(h_arrow_2) + crd[i])[None, :]
        return np.append(h1, h2, axis=0)

    if atype == 'ch3':
        upper_arrow = crd[k] - crd[j]
        upper_arrow /= np.linalg.norm(upper_arrow)
        h1 = -upper_arrow + crd[i]
        axes = crd[j] - crd[i]
        rotate_matrix = rotate_by_axis(axes, 2 * np.pi / 3)
        h2 = np.dot(rotate_matrix, h1-crd[i])
        h2 /= np.linalg.norm(h2)
        h2 += crd[i]
        rotate_matrix = rotate_by_axis(axes, 4 * np.pi / 3)
        h3 = np.dot(rotate_matrix, h1-crd[i])
        h3 /= np.linalg.norm(h3)
        h3 += crd[i]
        h12 = np.append(h1[None, :], h2[None, :], axis=0)
        return np.append(h12, h3[None, :], axis=0)

    if atype == 'cc3':
        h1 = crd[k]
        upper_arrow = crd[j] - crd[i]
        rotate_matrix = rotate_by_axis(upper_arrow, 2 * np.pi / 3)
        h2 = np.dot(rotate_matrix, h1-crd[i])
        h2 /= np.linalg.norm(h2)
        return (h2 + crd[i])[None, :]

    if atype == 'c2h2':
        right_arrow = crd[k] - crd[i]
        rotate_matrix = rotate_by_axis(right_arrow, 2 * np.pi / 3)
        h1 = np.dot(rotate_matrix, crd[j]-crd[i])
        h2 = np.dot(rotate_matrix, h1)
        h1 /= np.linalg.norm(h1)
        h1 = (h1 + crd[i])[None, :]
        h2 /= np.linalg.norm(h2)
        h2 = (h2 + crd[i])[None, :]
        return np.append(h1, h2, axis=0)

    if atype == 'h2o':
        if i is None:
            raise ValueError('The index of O atom should be given.')

    return None
