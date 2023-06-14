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
"""test"""
import math

import numpy as np

from mindspore import ops

from mindchemistry.e3 import identity_angles, rand_angles, compose_angles, matrix_x, matrix_y, matrix_z, \
    angles_to_matrix, matrix_to_angles, xyz_to_angles, angles_to_xyz

PI = math.pi
np.random.seed(123)


def test_unitary():
    angles = rand_angles(2, 1)
    rot = angles_to_matrix(*angles)
    perm = tuple(range(len(rot.asnumpy().shape)))
    perm = perm[:-2] + (perm[-1],) + (perm[-2],)
    assert np.allclose(np.matmul(rot.asnumpy(), rot.asnumpy().transpose(
        perm)), angles_to_matrix(*identity_angles(2, 1)).asnumpy(), rtol=1e-3, atol=1e-6)


def test_conversions():
    angles = rand_angles(2, 3)
    angles_new = matrix_to_angles(angles_to_matrix(*angles))
    for a, a_new in zip(angles, angles_new):
        assert np.allclose(a.asnumpy(), a_new.asnumpy() % (2 * PI), rtol=1e-3, atol=1e-6)

    ab = angles[:2]
    ab_new = xyz_to_angles(angles_to_xyz(*ab))
    for a, a_new in zip(ab, ab_new):
        assert np.allclose(a.asnumpy(), a_new.asnumpy() % (2 * PI), rtol=1e-3, atol=1e-6)


def test_compose():
    rot_x = matrix_x(PI / 2)
    rot_y = matrix_y(PI / 2)
    rot_z = matrix_z(-PI / 2)
    assert np.allclose(ops.matmul(rot_y, rot_x).asnumpy(), ops.matmul(
        rot_x, rot_z).asnumpy(), rtol=1e-3, atol=1e-6)

    angles1 = rand_angles(3)
    angles2 = rand_angles(3)
    rot = ops.matmul(angles_to_matrix(*angles1), angles_to_matrix(*angles2))
    rot_compose = angles_to_matrix(*compose_angles(*angles1, *angles2))
    assert np.allclose(rot.asnumpy(), rot_compose.asnumpy(),
                       rtol=1e-3, atol=1e-6)


if __name__ == '__main__':
    test_unitary()
    test_conversions()
    test_compose()
