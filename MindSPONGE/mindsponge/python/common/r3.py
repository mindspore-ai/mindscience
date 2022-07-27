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
"""r3"""

import numpy as np
import mindspore.numpy as mnp


def vecs_sub(v1, v2):
    """Computes v1 - v2."""
    return v1 - v2


def vecs_robust_norm(v, epsilon=1e-8, use_numpy=False):
    """Computes norm of vectors 'v'."""
    if use_numpy:
        return np.sqrt(np.square(v[0]) + np.square(v[1]) + np.square(v[2]) + epsilon)
    return mnp.sqrt(mnp.sum(mnp.square(v), axis=-1) + epsilon)


def vecs_robust_normalize(v, epsilon=1e-8, use_numpy=False):
    """Normalizes vectors 'v'."""

    norms = vecs_robust_norm(v, epsilon, use_numpy=use_numpy)
    if use_numpy:
        return v / norms[None, ...]
    return v / norms[..., None]


def vecs_dot_vecs(v1, v2, use_numpy=False):
    """Dot product of vectors 'v1' and 'v2'."""
    if use_numpy:
        return np.sum(v1 * v2, axis=0)
    return mnp.sum(v1 * v2, axis=-1)


def vecs_cross_vecs(v1, v2, use_numpy=False):
    """Cross product of vectors 'v1' and 'v2'."""
    if use_numpy:
        out = np.array((v1[1] * v2[2] - v1[2] * v2[1],
                        v1[2] * v2[0] - v1[0] * v2[2],
                        v1[0] * v2[1] - v1[1] * v2[0]))
        return out
    return mnp.concatenate(((v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1])[..., None],
                            (v1[..., 2] * v2[..., 0] -
                             v1[..., 0] * v2[..., 2])[..., None],
                            (v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0])[..., None]), axis=-1)


def rots_from_two_vecs(e0_unnormalized, e1_unnormalized, use_numpy=False):
    """Create rotation matrices from unnormalized vectors for the x and y-axes."""

    # Normalize the unit vector for the x-axis, e0.
    e0 = vecs_robust_normalize(e0_unnormalized, use_numpy=use_numpy)

    # make e1 perpendicular to e0.
    c = vecs_dot_vecs(e1_unnormalized, e0, use_numpy=use_numpy)
    if use_numpy:
        e1 = e1_unnormalized - c[None, ...] * e0
    else:
        e1 = e1_unnormalized - c[..., None] * e0
    e1 = vecs_robust_normalize(e1, use_numpy=use_numpy)

    # Compute e2 as cross product of e0 and e1.
    e2 = vecs_cross_vecs(e0, e1, use_numpy=use_numpy)

    if use_numpy:
        rots = np.array((e0[0], e1[0], e2[0], e0[1],
                         e1[1], e2[1], e0[2], e1[2], e2[2]))
        return rots

    rots = mnp.concatenate(
        (mnp.concatenate([e0[..., 0][None, ...], e1[..., 0][None, ...], e2[..., 0][None, ...]], axis=0)[None, ...],
         mnp.concatenate([e0[..., 1][None, ...], e1[..., 1]
                          [None, ...], e2[..., 1][None, ...]], axis=0)[None, ...],
         mnp.concatenate([e0[..., 2][None, ...], e1[..., 2][None, ...], e2[..., 2][None, ...]], axis=0)[None, ...]),
        axis=0)
    return rots


def rigids_from_3_points(
        point_on_neg_x_axis,  # shape (...)
        origin,  # shape (...)
        point_on_xy_plane,  # shape (...)
        use_numpy=False
):  # shape (...)
    """Create Rigids from 3 points. """
    m = rots_from_two_vecs(
        e0_unnormalized=vecs_sub(origin, point_on_neg_x_axis),
        e1_unnormalized=vecs_sub(point_on_xy_plane, origin),
        use_numpy=use_numpy)
    return m, origin


def invert_rots(m, use_numpy=False):
    """Computes inverse of rotations 'm'."""
    if use_numpy:
        out = np.array((m[0], m[3], m[6],
                        m[1], m[4], m[7],
                        m[2], m[5], m[8]))
        return out
    return mnp.transpose(m, (1, 0, 2, 3, 4))


def rots_mul_vecs(m, v, use_numpy=False):
    """Apply rotations 'm' to vectors 'v'."""
    if use_numpy:
        out = np.array((m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
                        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
                        m[6] * v[0] + m[7] * v[1] + m[8] * v[2]))
        return out
    return mnp.concatenate(((m[0][0] * v[..., 0] + m[0][1] * v[..., 1] + m[0][2] * v[..., 2])[..., None],
                            (m[1][0] * v[..., 0] + m[1][1] *
                             v[..., 1] + m[1][2] * v[..., 2])[..., None],
                            (m[2][0] * v[..., 0] + m[2][1] * v[..., 1] + m[2][2] * v[..., 2])[..., None]), axis=-1)


def invert_rigids(rot, trans, use_numpy=False):
    """Computes group inverse of rigid transformations 'r'."""
    inv_rots = invert_rots(rot, use_numpy=use_numpy)
    t = rots_mul_vecs(inv_rots, trans, use_numpy=use_numpy)
    inv_trans = -t
    return inv_rots, inv_trans


def vecs_add(v1, v2):
    """Add two vectors 'v1' and 'v2'."""
    return v1 + v2


def rigids_mul_vecs(rot, trans, v, use_numpy=False):
    """Apply rigid transforms 'r' to points 'v'."""
    return vecs_add(rots_mul_vecs(rot, v, use_numpy=use_numpy), trans)


def rigids_mul_rots(x, y, use_numpy=False):
    """numpy version of getting results rigids x multiply rots y"""
    rigids = (rots_mul_rots(x[0], y, use_numpy=use_numpy), x[1])
    return rigids


def rots_mul_rots(x, y, use_numpy=False):
    """numpy version of getting result of rots x multiply rots y"""
    vecs0 = rots_mul_vecs(x, (y[0], y[3], y[6]), use_numpy)
    vecs1 = rots_mul_vecs(x, (y[1], y[4], y[7]), use_numpy)
    vecs2 = rots_mul_vecs(x, (y[2], y[5], y[8]), use_numpy)
    return np.array((vecs0[0], vecs1[0], vecs2[0], vecs0[1], vecs1[1], vecs2[1], vecs0[2], vecs1[2], vecs2[2]))


def vecs_from_tensor(inputs):
    """get vectors from input tensor"""
    num_components = inputs.shape[-1]
    assert num_components == 3
    return np.array((inputs[..., 0], inputs[..., 1], inputs[..., 2]))


def rigids_to_tensor_flat12(inputs):  # outputs shape (..., 12)
    """transfer rigids to flat tensor"""
    return np.stack(list(inputs[0]) + list(inputs[1]), axis=-1)


def rots_from_tensor3x3(inputs):
    """get rotation from inputs tensor"""
    assert inputs.shape[-1] == 3
    assert inputs.shape[-2] == 3
    return np.array((inputs[..., 0, 0], inputs[..., 0, 1], inputs[..., 0, 2],
                     inputs[..., 1, 0], inputs[..., 1, 1], inputs[..., 1, 2],
                     inputs[..., 2, 0], inputs[..., 2, 1], inputs[..., 2, 2]))
