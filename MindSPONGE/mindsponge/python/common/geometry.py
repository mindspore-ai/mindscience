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
import numpy as np
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.ops import operations as P

QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, -1]]

QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, -1, 0]]

QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0],
                          [0, 0, 0, -1],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0]]

QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, -1, 0, 0],
                          [1, 0, 0, 0]]

QUAT_MULTIPLY_BY_VEC = Tensor(QUAT_MULTIPLY[:, 1:, :])

QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[0, 0, 2], [0, 0, 0], [2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[0, 0, 0], [0, 0, 2], [0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[0, 0, 0], [0, 0, -2], [0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[0, 0, 2], [0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[0, -2, 0], [2, 0, 0], [0, 0, 0]]  # kr

QUAT_TO_ROT = Tensor(QUAT_TO_ROT)


def vecs_scale(v, scale):
    """vec scale"""
    scaled_vecs = (v[0] * scale, v[1] * scale, v[2] * scale)
    return scaled_vecs


def rots_scale(rot, scale):
    """rots scale"""
    scaled_rots = (rot[0] * scale, rot[1] * scale, rot[2] * scale,
                   rot[3] * scale, rot[4] * scale, rot[5] * scale,
                   rot[6] * scale, rot[7] * scale, rot[8] * scale)
    return scaled_rots


def vecs_sub(v1, v2):
    """Computes v1 - v2."""
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def vecs_robust_norm(v, epsilon=1e-8):
    """Computes norm of vectors 'v'."""
    v_l2_norm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + epsilon
    v_norm = v_l2_norm ** 0.5
    return v_norm


def vecs_robust_normalize(v, epsilon=1e-8):
    """Normalizes vectors 'v'."""
    norms = vecs_robust_norm(v, epsilon)
    return (v[0] / norms, v[1] / norms, v[2] / norms)


def vecs_dot_vecs(v1, v2):
    """Dot product of vectors 'v1' and 'v2'."""
    res = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    return res


def vecs_cross_vecs(v1, v2):
    """Cross product of vectors 'v1' and 'v2'."""
    cross_res = (v1[1] * v2[2] - v1[2] * v2[1],
                 v1[2] * v2[0] - v1[0] * v2[2],
                 v1[0] * v2[1] - v1[1] * v2[0])
    return cross_res


def rots_from_two_vecs(e0_unnormalized, e1_unnormalized):
    """Create rotation matrices from unnormalized vectors for the x and y-axes."""

    # Normalize the unit vector for the x-axis, e0.
    e0 = vecs_robust_normalize(e0_unnormalized)

    # make e1 perpendicular to e0.
    c = vecs_dot_vecs(e1_unnormalized, e0)
    e1 = vecs_sub(e1_unnormalized, vecs_scale(e0, c))
    e1 = vecs_robust_normalize(e1)

    # Compute e2 as cross product of e0 and e1.
    e2 = vecs_cross_vecs(e0, e1)
    rots = (e0[0], e1[0], e2[0],
            e0[1], e1[1], e2[1],
            e0[2], e1[2], e2[2])
    return rots


def rigids_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane):
    """Create Rigids from 3 points. """
    m = rots_from_two_vecs(
        e0_unnormalized=vecs_sub(origin, point_on_neg_x_axis),
        e1_unnormalized=vecs_sub(point_on_xy_plane, origin))
    rigid = (m, origin)
    return rigid


def invert_rots(m):
    """Computes inverse of rotations 'm'."""
    invert = (m[0], m[3], m[6],
              m[1], m[4], m[7],
              m[2], m[5], m[8])
    return invert


def rots_mul_vecs(m, v):
    """Apply rotations 'm' to vectors 'v'."""
    out = (m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
           m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
           m[6] * v[0] + m[7] * v[1] + m[8] * v[2])
    return out


def invert_rigids(rigids):
    """Computes group inverse of rigid transformations 'r'."""
    rot, trans = rigids
    inv_rots = invert_rots(rot)
    t = rots_mul_vecs(inv_rots, trans)
    inv_trans = (-1.0 * t[0], -1.0 * t[1], -1.0 * t[2])
    inv_rigids = (inv_rots, inv_trans)
    return inv_rigids


def vecs_add(v1, v2):
    """Add two vectors 'v1' and 'v2'."""
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


def rigids_mul_vecs(rigids, v):
    """Apply rigid transforms 'r' to points 'v'."""
    return vecs_add(rots_mul_vecs(rigids[0], v), rigids[1])


def rigids_mul_rots(x, y):
    """numpy version of getting results rigids x multiply rots y"""
    rigids = (rots_mul_rots(x[0], y), x[1])
    return rigids


def rigids_mul_rigids(a, b):
    """rigids mul rigids"""
    rot = rots_mul_rots(a[0], b[0])
    trans = vecs_add(a[1], rots_mul_vecs(a[0], b[1]))
    return (rot, trans)


def rots_mul_rots(x, y):
    """numpy version of getting result of rots x multiply rots y"""
    vecs0 = rots_mul_vecs(x, (y[0], y[3], y[6]))
    vecs1 = rots_mul_vecs(x, (y[1], y[4], y[7]))
    vecs2 = rots_mul_vecs(x, (y[2], y[5], y[8]))
    rots = (vecs0[0], vecs1[0], vecs2[0], vecs0[1], vecs1[1], vecs2[1], vecs0[2], vecs1[2], vecs2[2])
    return rots


def vecs_from_tensor(inputs):
    """get vectors from input tensor"""
    num_components = inputs.shape[-1]
    assert num_components == 3
    return (inputs[..., 0], inputs[..., 1], inputs[..., 2])


def vecs_to_tensor(v):
    """Converts 'v' to tensor with shape 3, inverse of 'vecs_from_tensor'."""
    return mnp.stack([v[0], v[1], v[2]], axis=-1)


def make_transform_from_reference(point_a, point_b, point_c):
    '''construct rotation and translation from given points
    calculate the rotation matrix and translation meets:
    a) point_b is the original point
    b) point_c is on the x_axis
    c) the plane a-b-c is on the x-y plane
    Return:
    rotation: Tuple [xx, xy, xz, yx, yy, yz, zx, zy, zz]
    translation: Tuple [x, y, z]
    '''

    # step 1 : shift the crd system by -point_b (point_b is the origin)
    translation = -point_b
    point_c = point_c + translation
    point_a = point_a + translation
    # step 2: rotate the crd system around z-axis to put point_c on x-z plane
    c_x, c_y, c_z = vecs_from_tensor(point_c)
    sin_c1 = -c_y / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2)
    cos_c1 = c_x / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2)
    zeros = mnp.zeros_like(sin_c1)
    ones = mnp.ones_like(sin_c1)
    c1_rot_matrix = (cos_c1, -sin_c1, zeros,
                     sin_c1, cos_c1, zeros,
                     zeros, zeros, ones)
    # step 2 : rotate the crd system around y_axis to put point_c on x-axis
    sin_c2 = c_z / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2 + c_z ** 2)
    cos_c2 = mnp.sqrt(c_x ** 2 + c_y ** 2) / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2 + c_z ** 2)
    c2_rot_matrix = (cos_c2, zeros, sin_c2,
                     zeros, ones, zeros,
                     -sin_c2, zeros, cos_c2)
    c_rot_matrix = rots_mul_rots(c2_rot_matrix, c1_rot_matrix)
    # step 3: rotate the crd system in y-z plane to put point_a in x-y plane
    vec_a = vecs_from_tensor(point_a)
    _, rotated_a_y, rotated_a_z = rots_mul_vecs(c_rot_matrix, vec_a)

    sin_n = -rotated_a_z / mnp.sqrt(1e-20 + rotated_a_y ** 2 + rotated_a_z ** 2)
    cos_n = rotated_a_y / mnp.sqrt(1e-20 + rotated_a_y ** 2 + rotated_a_z ** 2)
    a_rot_matrix = (ones, zeros, zeros,
                    zeros, cos_n, -sin_n,
                    zeros, sin_n, cos_n)
    rotation_matrix = rots_mul_rots(a_rot_matrix, c_rot_matrix)
    translation = point_b
    translation = vecs_from_tensor(translation)
    return rotation_matrix, translation


def rots_from_tensor(rots, use_numpy=False):
    """rots from tensor"""
    if use_numpy:
        rots = np.reshape(rots, rots.shape[:-2] + (9,))
    else:
        rots = P.Reshape()(rots, P.Shape()(rots)[:-2] + (9,))
    rotation = (rots[..., 0], rots[..., 1], rots[..., 2],
                rots[..., 3], rots[..., 4], rots[..., 5],
                rots[..., 6], rots[..., 7], rots[..., 8])
    return rotation


def rots_to_tensor(rots, use_numpy=False):
    """rots to tensor"""
    assert len(rots) == 9
    if use_numpy:
        rots = np.stack(rots, axis=-1)
        rots = np.reshape(rots, rots.shape[:-1] + (3, 3))
    else:
        rots = mnp.stack(rots, axis=-1)
        rots = mnp.reshape(rots, rots.shape[:-1] + (3, 3))
    return rots


def rot_to_quat(rot, stack=False, use_numpy=False):
    """rot to quat"""
    if stack:
        rot = rots_from_tensor(rot, use_numpy)
    xx, xy, xz, yx, yy, yz, zx, zy, zz = rot

    quaternion = (1. / 3.) * mnp.stack((xx + yy + zz, zy - yz, xz - zx, yx - xy), axis=-1)
    return quaternion


def quat_affine(quaternion, translation, rotation=None, normalize=True, unstack_inputs=False, use_numpy=False):
    """create quat affine representations"""
    if unstack_inputs:
        if rotation is not None:
            rotation = rots_from_tensor(rotation, use_numpy)
        translation = vecs_from_tensor(translation)

    if normalize and quaternion is not None:
        quaternion = quaternion / mnp.norm(quaternion, axis=-1, keepdims=True)
    if rotation is None:
        rotation = quat_to_rot(quaternion)
    return quaternion, rotation, translation


def quat_to_rot(normalized_quat, use_numpy=False):
    """Convert a normalized quaternion to a rotation matrix."""
    if use_numpy:
        rot_tensor = np.sum(np.reshape(QUAT_TO_ROT.asnumpy(), (4, 4, 9)) * normalized_quat[..., :, None, None] \
                * normalized_quat[..., None, :, None], axis=(-3, -2))
        rot_tensor = rots_from_tensor(rot_tensor, use_numpy)
    else:
        rot_tensor = mnp.sum(mnp.reshape(QUAT_TO_ROT, (4, 4, 9)) * normalized_quat[..., :, None, None] *
                             normalized_quat[..., None, :, None], axis=(-3, -2))
        rot_tensor = P.Split(-1, 9)(rot_tensor)
        rot_tensor = (P.Squeeze()(rot_tensor[0]), P.Squeeze()(rot_tensor[1]), P.Squeeze()(rot_tensor[2]),
                      P.Squeeze()(rot_tensor[3]), P.Squeeze()(rot_tensor[4]), P.Squeeze()(rot_tensor[5]),
                      P.Squeeze()(rot_tensor[6]), P.Squeeze()(rot_tensor[7]), P.Squeeze()(rot_tensor[8]))
    return rot_tensor


def initial_affine(num_residues, use_numpy=False):
    """initial affine"""
    if use_numpy:
        quaternion = np.tile(np.reshape(np.asarray([1., 0., 0., 0.]), [1, 4]), [num_residues, 1])
        translation = np.zeros([num_residues, 3])
    else:
        quaternion = mnp.tile(mnp.reshape(mnp.asarray([1., 0., 0., 0.]), [1, 4]), [num_residues, 1])
        translation = mnp.zeros([num_residues, 3])
    return quat_affine(quaternion, translation, unstack_inputs=True, use_numpy=use_numpy)


def vecs_expend_dims(v, axis):
    """vecs expend dim"""
    v = (P.ExpandDims()(v[0], axis), P.ExpandDims()(v[1], axis), P.ExpandDims()(v[2], axis))
    return v


def rots_expend_dims(rots, axis):
    """rot expend dims"""
    rots = (P.ExpandDims()(rots[0], axis), P.ExpandDims()(rots[1], axis), P.ExpandDims()(rots[2], axis),
            P.ExpandDims()(rots[3], axis), P.ExpandDims()(rots[4], axis), P.ExpandDims()(rots[5], axis),
            P.ExpandDims()(rots[6], axis), P.ExpandDims()(rots[7], axis), P.ExpandDims()(rots[8], axis))
    return rots


def invert_point(transformed_point, rotation, translation, extra_dims=0, stack=False, use_numpy=False):
    """invert_point"""
    if stack:
        rotation = rots_from_tensor(rotation, use_numpy)
        translation = vecs_from_tensor(translation)
    for _ in range(extra_dims):
        rotation = rots_expend_dims(rotation, -1)
        translation = vecs_expend_dims(translation, -1)
    rot_point = vecs_sub(transformed_point, translation)
    return rots_mul_vecs(invert_rots(rotation), rot_point)


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""

    return mnp.sum(QUAT_MULTIPLY_BY_VEC * quat[..., :, None, None] * vec[..., None, :, None],
                   axis=(-3, -2))


def pre_compose(quaternion, rotation, translation, update):
    """Return a new QuatAffine which applies the transformation update first.

    Args:
    update: Length-6 vector. 3-vector of x, y, and z such that the quaternion
    update is (1, x, y, z) and zero for the 3-vector is the identity
    quaternion. 3-vector for translation concatenated.

    Returns:
    quaternion: [..., 4]
    rotation: Tuple [xx, xy, xz, yx, yy, yz, zx, zy, zz]
    translation: Tuple [x, y, z]
    """

    vector_quaternion_update, x, y, z = mnp.split(update, [3, 4, 5], axis=-1)
    trans_update = [mnp.squeeze(x, axis=-1), mnp.squeeze(y, axis=-1), mnp.squeeze(z, axis=-1)]
    new_quaternion = (quaternion + quat_multiply_by_vec(quaternion, vector_quaternion_update))
    rotated_trans_update = rots_mul_vecs(rotation, trans_update)
    new_translation = vecs_add(translation, rotated_trans_update)
    return quat_affine(new_quaternion, new_translation)


def quaternion_to_tensor(quaternion, translation):
    """quaternion to tensor"""
    translation = (P.ExpandDims()(translation[0], -1), P.ExpandDims()(translation[1], -1),
                   P.ExpandDims()(translation[2], -1),)
    return mnp.concatenate((quaternion,) + translation, axis=-1)


def quaternion_from_tensor(tensor, normalize=False):
    """quaternion from tensor"""
    quaternion, tx, ty, tz = mnp.split(tensor, [4, 5, 6], axis=-1)
    translation = (P.Squeeze()(tx), P.Squeeze()(ty), P.Squeeze()(tz))
    return quat_affine(quaternion, translation, normalize=normalize)


def apply_to_point(rotation, translation, point, extra_dims=0):
    """apply to point func"""
    for _ in range(extra_dims):
        rotation = rots_expend_dims(rotation, -1)
        translation = vecs_expend_dims(translation, -1)
    rot_point = rots_mul_vecs(rotation, point)
    result = vecs_add(rot_point, translation)
    return result
