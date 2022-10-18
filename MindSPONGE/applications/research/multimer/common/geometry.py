# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""Geometry"""
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindsponge.common.geometry import vecs_dot_vecs, vecs_sub, vecs_cross_vecs, \
    rots_expend_dims, vecs_expend_dims, invert_rigids, rigids_mul_vecs, \
        vecs_from_tensor, vecs_scale


def rots_mul_rots(r1, r2):
    """rots_mul_rots."""
    out = (r1[0] * r2[0], r1[1] * r2[1], r1[2] * r2[2],
           r1[3] * r2[3], r1[4] * r2[4], r1[5] * r2[5],
           r1[6] * r2[6], r1[7] * r2[7], r1[8] * r2[8])
    return out


def trans_mul_trans(t1, t2):
    """trans_mul_trans."""
    out = (t1[0] * t2[0], t1[1] * t2[1], t1[2] * t2[2])
    return out


def multimer_vecs_robust_norm(v, epsilon=1e-6):
    """multime computes norm of vectors 'v'."""
    v_l2_norm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    if epsilon:
        v_l2_norm = mnp.maximum(v_l2_norm, epsilon**2)
    return mnp.sqrt(v_l2_norm)


def multimer_vecs_robust_normalize(v, epsilon=1e-6):
    """multimer normalizes vectors 'v'."""
    norms = multimer_vecs_robust_norm(v, epsilon)
    return (v[0] / norms, v[1] / norms, v[2] / norms)


def multimer_rots_from_two_vecs(e0_unnormalized, e1_unnormalized):
    """multimer_rots_from_two_vecs."""
    e0 = multimer_vecs_robust_normalize(e0_unnormalized)
    c = vecs_dot_vecs(e1_unnormalized, e0)
    e1 = vecs_sub(e1_unnormalized, vecs_scale(e0, c))
    e1 = multimer_vecs_robust_normalize(e1)
    e2 = vecs_cross_vecs(e0, e1)

    rots = (e0[0], e1[0], e2[0],
            e0[1], e1[1], e2[1],
            e0[2], e1[2], e2[2])
    return rots


def multimer_rigids_from_3_points(vec_a, vec_b, vec_c):
    """Create multimer Rigids from 3 points. """
    m = multimer_rots_from_two_vecs(
        e0_unnormalized=vecs_sub(vec_c, vec_b),
        e1_unnormalized=vecs_sub(vec_a, vec_b))
    rigid = (m, vec_b)
    return rigid


def multimer_rigids_get_unit_vector(point_a, point_b, point_c):
    """multimer_rigids_get_unit_vector."""
    rigid = multimer_rigids_from_3_points(vecs_from_tensor(point_a),
                                          vecs_from_tensor(point_b),
                                          vecs_from_tensor(point_c))
    rot, trans = rigid
    rotation = rots_expend_dims(rot, -1)
    translation = vecs_expend_dims(trans, -1)
    inv_rigid = invert_rigids((rotation, translation))
    rigid_vec = rigids_mul_vecs(inv_rigid, vecs_expend_dims(trans, -2))
    unit_vector = multimer_vecs_robust_normalize(rigid_vec)
    return unit_vector


def multimer_rigids_compute_dihedral_angle(a, b, c, d):
    """multimer_rigids_compute_dihedral_angle."""
    v1 = vecs_sub(a, b)
    v2 = vecs_sub(b, c)
    v3 = vecs_sub(d, c)

    c1 = vecs_cross_vecs(v1, v2)
    c2 = vecs_cross_vecs(v3, v2)
    c3 = vecs_cross_vecs(c2, c1)

    v2_mag = multimer_vecs_robust_norm(v2)
    return mnp.arctan2(vecs_dot_vecs(c3, v2), v2_mag * vecs_dot_vecs(c1, c2))


def multimer_from_quaternion(w, x, y, z, normalize=True, epsilon=1e-6):
    """multimer_from_quaternion."""
    if normalize:
        inv_norm = P.Rsqrt()(mnp.maximum(epsilon, w**2 + x**2 + y**2 + z**2))
        w *= inv_norm
        x *= inv_norm
        y *= inv_norm
        z *= inv_norm
    xx = 1 - 2 * (mnp.square(y) + mnp.square(z))
    xy = 2 * (x * y - w * z)
    xz = 2 * (x * z + w * y)
    yx = 2 * (x * y + w * z)
    yy = 1 - 2 * (mnp.square(x) + mnp.square(z))
    yz = 2 * (y * z - w * x)
    zx = 2 * (x * z - w * y)
    zy = 2 * (y * z + w * x)
    zz = 1 - 2 * (mnp.square(x) + mnp.square(y))
    rots = (xx, xy, xz,
            yx, yy, yz,
            zx, zy, zz)
    return rots


def multimer_square_euclidean_distance(v1, v2, epsilon):
    """multimer_square_euclidean_distance."""
    difference = vecs_sub(v1, v2)
    distance = vecs_dot_vecs(difference, difference)
    if epsilon:
        distance = mnp.maximum(distance, epsilon)
    return distance
