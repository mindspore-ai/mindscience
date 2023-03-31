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
    r"""
    Scale the vector.

    .. math::
        \begin{split}
        &v=(x1,x2,x3) \\
        &scaled\_{vecs} = (scale*x1,scale*x2,scale*x3) \\
        \end{split}

    Args:
        v(Tuple):       Vector will be scaled, :math:`(x,y,z)`. x, y, z are scalars or Tensor with same shape.
        scale(float):   Value of scale.

    Returns:
        Tuple with length of 3, vector after scaled with the same shape as input v.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from mindsponge.common.geometry import vecs_scale
        >>> x= Tensor(np.ones(256), mstype.float32)
        >>> y= Tensor(np.ones(256), mstype.float32)
        >>> z= Tensor(np.ones(256), mstype.float32)
        >>> scale=10
        >>> result=vecs_scale((x,y,z),scale)
        >>> print(len(result))
        >>> print(result[0].shape)
        >>> print(result[1].shape)
        >>> print(result[2].shape)
        3
        (256,)
        (256,)
        (256,)
    """
    scaled_vecs = (v[0] * scale, v[1] * scale, v[2] * scale)
    return scaled_vecs


def rots_scale(rot, scale):
    r"""
    Scaling of rotation matrixs.

    .. math::
        \begin{split}
        &rot=(xx,xy,xz,yx,yy,yz,zx,zy,zz) \\
        &scaled\_{rots} = (scale*xx,scale*xy,scale*xz,scale*yx,scale*yy,scale*yz,scale*zx,scale*zy,scale*zz)
        \end{split}

    Args:
        rot(Tuple):     Rots, length is 9, :math:`(xx,xy,xz,yx,yy,yz,zx,zy,zz)` . Data type is scalar or
                        Tensor with the same shape.
        scale(float):   Value of scale.

    Returns:
        Tuple, scaled rotation matrixs. Length is 9, shape is the same as the input rots' shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from mindsponge.common.geometry import rots_scale
        >>> x = Tensor(np.ones(256), mstype.float32)
        >>> result = rots_scale((x, x, x, x, x, x, x, x, x),10)
        >>> print(len(result))
        >>> print(result[0].shape)
        >>> print(result[1].shape)
        >>> print(result[2].shape)
        >>> print(result[3].shape)
        >>> print(result[4].shape)
        >>> print(result[5].shape)
        >>> print(result[6].shape)
        >>> print(result[7].shape)
        >>> print(result[8].shape)
        3
        (256,)
        (256,)
        (256,)
        (256,)
        (256,)
        (256,)
        (256,)
        (256,)
        (256,)
    """
    scaled_rots = (rot[0] * scale, rot[1] * scale, rot[2] * scale,
                   rot[3] * scale, rot[4] * scale, rot[5] * scale,
                   rot[6] * scale, rot[7] * scale, rot[8] * scale)
    return scaled_rots


def vecs_sub(v1, v2):
    r"""
    Subtract two vectors.

    .. math::
        \begin{split}
        &v1=(x1,x2,x3) \\
        &v2=(x1',x2',x3') \\
        &result=(x1-x1',x2-x2',x3-x3') \\
        \end{split}

    Args:
        v1(Tuple):  input vector 1 :math:`(x, y, z)`, data type is scalar or Tensor with same shape.
        v2(Tuple):  input vector 2 :math:`(x, y, z)`, data type is scalar or Tensor with same shape.

    Returns:
        Tuple. Length is 3, :math:`(x', y', z')` , data type is scalar or Tensor with same shape as v1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from mindsponge.common.geometry import vecs_sub
        >>> x= Tensor(np.ones(256), mstype.float32)
        >>> y= Tensor(np.ones(256), mstype.float32)
        >>> z= Tensor(np.ones(256), mstype.float32)
        >>> result=vecs_sub((x,y,z),(x,y,z))
        >>> print(len(result))
        >>> print(result[0].shape)
        >>> print(result[1].shape)
        >>> print(result[2].shape)
        3
        (256,)
        (256,)
        (256,)
    """
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def vecs_robust_norm(v, epsilon=1e-8):
    r"""
    Calculate the l2-norm of a vector.

    .. math::
        \begin{split}
        &v=(x1,x2,x3) \\
        &l2\_norm=\sqrt{x1*x1+x2*x2+x3*x3+epsilon} \\
        \end{split}

    Args:
        v(Tuple):       Input vector :math:`(x,y,z)` . Data type is scalar or Tensor with same shape.
        epsilon(float): A very small number to prevent the result from being 0. Default: 1e-8.

    Returns:
        Tensor, 2-Norm calculated by vector v. Shape is the same as v.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from mindsponge.common.geometry import vecs_robust_norm
        >>> x= Tensor(np.ones(256), mstype.float32)
        >>> y= Tensor(np.ones(256), mstype.float32)
        >>> z= Tensor(np.ones(256), mstype.float32)
        >>> result=vecs_robust_norm((x,y,z))
        >>> print(result.shape)
        (256)
    """
    v_l2_norm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + epsilon
    v_norm = v_l2_norm ** 0.5
    return v_norm


def vecs_robust_normalize(v, epsilon=1e-8):
    r"""
    Use l2-norm normalization vectors

    .. math::
        \begin{split}
        &v=(x1,x2,x3) \\
        &l2\_norm=\sqrt{x1*x1+x2*x2+x3*x3+epsilon} \\
        &result=(x1/l2\_norm, x2/l2\_norm, x3/l2\_norm) \\
        \end{split}

    Args:
        v(Tuple):       Input vector :math:`(x,y,z)` . Data type is scalar or Tensor with same shape.
        epsilon(float): Minimal value, prevent the result from being 0. Default: 1e-8.

    Returns:
        Tuple with length of 3, normalized 2-Norm calculated by vector v. Shape is the same as v.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from mindsponge.common.geometry import vecs_robust_normalize
        >>> x= Tensor(np.ones(256), mstype.float32)
        >>> y= Tensor(np.ones(256), mstype.float32)
        >>> z= Tensor(np.ones(256), mstype.float32)
        >>> result=vecs_robust_normalize((x,y,z))
        >>> print(len(result))
        >>> print(result[0].shape)
        >>> print(result[1].shape)
        >>> print(result[2].shape)
            3
        (256,)
        (256,)
        (256,)
    """
    norms = vecs_robust_norm(v, epsilon)
    return (v[0] / norms, v[1] / norms, v[2] / norms)


def vecs_dot_vecs(v1, v2):
    r"""
    Dot product of vectors :math:`v_1 = (x_1, x_2, x_3)` and :math:`v_2 = (y_1, y_2, y_3)`.

    .. math::
        res = x_1 * y_1 + x_2 * y_2 + x_3 * y_3

    Args:
        v1 (tuple): vectors :math:`\vec v_1` , length is 3.
                    Data type is constant or Tensor with same shape.
        v2 (tuple): vectors :math:`\vec v_2` , length is 3.
                    Data type is constant or Tensor with same shape.

    Returns:
        float or Tensor with the same shape as the Tensor in input, dot product result of two vectors .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> v1 = (1, 2, 3)
        >>> v2 = (3, 4, 5)
        >>> ans = mindsponge.common.vecs_dot_vecs(v1, v2)
        >>> print(ans)
        26
    """
    res = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    return res


def vecs_cross_vecs(v1, v2):
    r"""
    Cross product of vectors :math:`v_1 = (x_1, x_2, x_3)` and :math:`v_2 = (y_1, y_2, y_3)`.

    .. math::
        cross_{res} = (x_2 * y_3 - x_3 * y_2, x_3 * y_1 - x_1 * y_3, x_1 * y_2 - x_2 * y_1)

    Args:
        v1 (tuple): vectors :math:`\vec v_1` , length is 3.
                    Data type is constant or Tensor with same shape.
        v2 (tuple): vectors :math:`\vec v_2` , length is 3.
                    Data type is constant or Tensor with same shape.

    Returns:
        tuple, cross product result of two vectors, length is 3.
            Data type is constant or Tensor with same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> v1 = (1, 2, 3)
        >>> v2 = (3, 4, 5)
        >>> ans = mindsponge.common.vecs_cross_vecs(v1, v2)
        >>> print(ans)
        (-2, 4, -2)
    """
    cross_res = (v1[1] * v2[2] - v1[2] * v2[1],
                 v1[2] * v2[0] - v1[0] * v2[2],
                 v1[0] * v2[1] - v1[1] * v2[0])
    return cross_res


def rots_from_two_vecs(e0_unnormalized, e1_unnormalized):
    r"""
    Put in two vectors :math:`\vec a = (a_x, a_y, a_z)` and :math:`\vec b = (b_x, b_y, b_z)`.
    Calculate the rotation matrix between local coordinate system, in which the x-y plane
    consists of two input vectors and global coordinate system.

    Calculate the unit vector :math:`\vec e_0 = \frac{\vec a}{|\vec a|}`
    as the unit vector of x axis.

    Then calculate the projected length of :math:`\vec b` on a axis.
    :math:`c = |\vec b| \cos\theta = \vec b \cdot \frac{\vec a}{|\vec a|}` .

    So the projected vector of :math:`b` on a axis is :math:`c\vec e_0`.
    The vector perpendicular to e0 is :math:`\vec e_1' = \vec b - c\vec e_0` .

    The unit vector of :math:`\vec e_1'` is :math:`\vec e_1 = \frac{\vec e_1'}{|\vec e_1'|}`,
    which is the y axis of the local coordinate system.

    Finally get the unit vector of z axis :math:`\vec e_2` by calculating cross product of
    :math:`\vec e_1` and :math:`\vec e_0`.

    The final rots is :math:`(e_{0x}, e_{1x}, e_{2x}, e_{0y}, e_{1y}, e_{2y}, e_{0z}, e_{1z}, e_{2z})`.

    Args:
        e0_unnormalized (tuple):    vectors :math:`\vec a` as x-axis of x-y plane,
                                    length is 3. Data type is constant or Tensor with same shape.
        e1_unnormalized (tuple):    vectors :math:`\vec b` forming x-y plane,
                                    length is 3. Data type is constant or Tensor with same shape.

    Returns:
        tuple, rotation matrix :math:`(e_{0x}, e_{1x}, e_{2x}, e_{0y}, e_{1y}, e_{2y}, e_{0z}, e_{1z}, e_{2z})` .
            Data type is constant or Tensor with same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> v1 = (1, 2, 3)
        >>> v2 = (3, 4, 5)
        >>> ans = mindsponge.common.rots_from_two_vecs(v1, v2)
        >>> print(ans)
        (0.4242640686695021, -0.808290367995452, 0.40824828617045156, 0.5656854248926695,
         -0.1154700520346678, -0.8164965723409039, 0.7071067811158369, 0.5773502639261153,
         0.4082482861704521)
    """

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
    r"""
    Gram-Schmidt process. Create rigids representation of 3 points local coordination system,
    point on negative x axis A, origin point O and point on x-y plane P.

    First calculate the coordinations of vector :math:`\vec AO` and :math:`\vec OP`. Then
    use `rots_from_two_vecs` get the rotation matrix.

    Distance between origin point O and the origin point of global coordinate system is
    the translations of rigid.

    Finally return the rotations and translations of rigid.

    Reference:
        `Jumper et al. (2021) Suppl. Alg. 21 'Gram-Schmidt process'
        <https://www.nature.com/articles/s41586-021-03819-2>`_.

    .. math::
        \begin{split}
        &\vec v_1 = \vec x_3 - \vec x_2 \\
        &\vec v_2 = \vec x_1 - \vec x_2 \\
        &\vec e_1 = \vec v_1 / ||\vec v_1|| \\
        &\vec u_2 = \vec v_2 - \vec  e_1(\vec e_1^T\vec v_2) \\
        &\vec e_2 = \vec u_2 / ||\vec u_2|| \\
        &\vec e_3 = \vec e_1 \times \vec e_2 \\
        &rotation = (\vec e_1, \vec e_2, \vec e_3) \\
        &translation = (\vec x_2) \\
        \end{split}

    Args:
        point_on_neg_x_axis (tuple):    point on negative x axis A, length is 3.
                                        Data type is constant or Tensor with same shape.
        origin (tuple):                 origin point O, length is 3.
                                        Data type is constant or Tensor with same shape.
        point_on_xy_plane (tuple):      point on x-y plane P, length is 3.
                                        Data type is constant or Tensor with same shape.

    Returns:
        tuple(rots, trans), rigid, length is 2. Include rots :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`
            and trans :math:`(x, y, z)` . Data type is constant or Tensor with same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> A = (1, 2, 3)
        >>> O = (4, 6, 8)
        >>> P = (5, 8, 11)
        >>> ans = mindsponge.common.rigids_from_3_points(A, O, P)
        >>> print(ans)
        ((0.4242640686695021, -0.808290367995452, 0.40824828617045156, 0.5656854248926695,
         -0.1154700520346678, -0.8164965723409039, 0.7071067811158369, 0.5773502639261153,
         0.4082482861704521), (4,6,8))
    """
    m = rots_from_two_vecs(
        e0_unnormalized=vecs_sub(origin, point_on_neg_x_axis),
        e1_unnormalized=vecs_sub(point_on_xy_plane, origin))
    rigid = (m, origin)
    return rigid


def invert_rots(m):
    r"""
    Computes inverse of rotations :math:`m`.

    rotations :math:`m = (xx, xy, xz, yx, yy, yz, zx, zy, zz)` and
    inverse of :math:`m` is :math:`m^{T} = (xx, yx, zx, xy, yy, zy, xz, yz, zz)` .

    Args:
        m (tuple):  rotations :math:`m` , length is 9.
                    Data type is constant or Tensor with same shape.

    Returns:
        tuple, inverse of rotations :math:`m` , length is 9. Data type is constant or Tensor with same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> m = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        >>> inv_m = mindsponge.common.invert_rots(m)
        >>> print(inv_m)
        (1, 4, 7, 2, 5, 8, 3, 6, 9)
    """
    invert = (m[0], m[3], m[6],
              m[1], m[4], m[7],
              m[2], m[5], m[8])
    return invert


def rots_mul_vecs(m, v):
    r"""
    Apply rotations :math:`\vec m = (m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8)`
    to vectors :math:`\vec v = (v_0, v_1, v_2)`.

    .. math::
        out = m \cdot v^T = (m_0 \times v_0 + m_1 \times v_1 + m_2 \times v_2,
                             m_3 \times v_0 + m_4 \times v_1 + m_5 \times v_2,
                             m_6 \times v_0 + m_7 \times v_1 + m_8 \times v_2)

    Args:
        m (tuple):  rotations :math:`\vec m` , length is 9.
                    Data type is constant or Tensor with same shape.
        v (tuple):  vectors :math:`\vec v` , length is 3.
                    Data type is constant or Tensor with same shape.

    Returns:
        tuple, vectors after rotations.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> m = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        >>> v = (1, 2, 3)
        >>> v1 = mindsponge.common.rots_mul_vecs(m, v)
        >>> print(v1)
        (14, 32, 50)
    """
    out = (m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
           m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
           m[6] * v[0] + m[7] * v[1] + m[8] * v[2])
    return out


def invert_rigids(rigids):
    r"""
    Computes group inverse of rigid transformations. Change rigid from
    local coordinate system to global coordinate system.

    Use `invert_rots` to calculate the invert rotations of rigid. Then use
    `rots_mul_vecs` to rotate the translations of rigid. The opposite of the
    result is the translations of invert rigid.

    .. math::
        \begin{split}
        &inv\_rots = r_r^T = (r_0, r_3, r_6, r_1, r_4, r_7, r_2, r_5, r_8) \\
        &inv\_trans = -r_r^T \cdot r_t^T = (- (r_0 \times t_0 + r_3 \times t_0 + r_6 \times t_0),
                                           - (r_1 \times t_1 + r_4 \times t_1 + r_7 \times t_1),
                                           - (r_2 \times t_2 + r_5 \times t_2 + r_8 \times t_2)) \\
        \end{split}

    Args:
        rigids (tuple): rigids, including the rots and trans changing rigids
                        from global coordinate system to local coordinate system.

    Returns:
        tuple(rots, trans), group inverse of rigid transformations, length is 2. Include rots
            :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` and trans :math:`(x, y, z)` .
            Data type is constant or Tensor with same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> a = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (3, 4, 5))
        >>> inv_a = mindsponge.common.invert_rigids(a)
        >>> print(inv_a)
        ((1, 4, 7, 2, 5, 8, 3, 6, 9), (-54.0, -66.0, -78.0))
    """
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
    r"""
    Transform vector :math:`\vec v` to rigid' local coordinate system.

    Multiply vector :math:`\vec v` and the rotations of rigid together
    and add the translations of rigid. The result is the output vector.

    .. math::
        v = r_rv+r_t

    Args:
        rigids (tuple): rigid.
        v (tuple):      vector :math:`\vec v` , length is 3. Data type is constant or Tensor with same shape.

    Returns:
        tuple, changed vector, length is 3. Data type is constant or Tensor with same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> a = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (3, 4, 5))
        >>> b = (1, 2, 3)
        >>> b1 = mindsponge.common.rigids_mul_vecs(a,b)
        >>> print(b1)
        (17, 36, 55)
    """
    return vecs_add(rots_mul_vecs(rigids[0], v), rigids[1])


def rigids_mul_rots(x, y):
    r"""
    Numpy version of getting results rigid :math:`x` multiply rotations :math:`\vec y` .

    Multiply rotations of rigid :math:`x[0]` with rotations :math:`\vec y`,
    the result is rigids new rotations. Translations of rigid will not changed.

    .. math::
        (r, t) = (x_ry, x_t)

    Args:
        x (tuple):  rigid :math:`x` . Length is 2. Include rots :math:`x_r = (xx, xy, xz, yx, yy, yz, zx, zy, zz)`
                    and trans :math:`x_t = (x, y, z)` . Data type is constant or Tensor with same shape.
        y (tuple):  rotations :math:`\vec y` , length is 9. Data type is constant or Tensor with same shape.

    Returns:
        tuple(rots, trans), length is 2, rigid whose rotations are changed.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> a = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (3, 4, 5))
        >>> b = (2, 3, 4, 1, 5, 6, 3, 8, 7)
        >>> b1 = mindsponge.common.rigids_mul_rots(a,b)
        >>> print(b1)
        ((13, 37, 37, 31, 85, 88, 49, 133, 139), (3, 4, 5))
    """
    rigids = (rots_mul_rots(x[0], y), x[1])
    return rigids


def rigids_mul_rigids(a, b):
    r"""
    Change rigid :math:`b` from its local coordinate system to rigid :math:`a`
    local coordinate system, using rigid rotations and translations.

    Use the rotations calculated by multiplying rotations of rigid :math:`b`
    and rigid :math:`a` as new rotations of rigid :math:`b` .

    Multiply the translations of rigid :math:`b` with rotations of rigid :math:`a` ,
    then add translations of rigid :math:`a` . The translations got is new translations
    of rigid :math:`b`.

    .. math::
        \begin{split}
        &r = a_rb_r \\
        &t = a_rb_t +a_t \\
        \end{split}

    Args:
        a (tuple):  rigid :math:`a` . Length is 2. Include rots :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`
                    and trans :math:`(x, y, z)` . Data type is constant or Tensor with same shape.
        b (tuple):  rigid :math:`b` . Length is 2. Include rots :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`
                    and trans :math:`(x, y, z)` . Data type is constant or Tensor with same shape.

    Returns:
        tuple(rots, trans), rigid :math:`b` changed. Length is 2.
            Include rots :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`
            and trans :math:`(x, y, z)` . Data type is constant or Tensor with same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> a = ((1, 2, 3, 4, 5, 6, 7, 8, 9), (3, 4, 5))
        >>> b = ((2, 3, 4, 1, 5, 6, 3, 8, 7), (1, 2, 3))
        >>> b1 = mindsponge.common.rigids_mul_rigids(a,b)
        >>> print(b1)
        ((13, 37, 37, 31, 85, 88, 49, 133, 139), (17, 36, 55))
    """
    rot = rots_mul_rots(a[0], b[0])
    trans = vecs_add(a[1], rots_mul_vecs(a[0], b[1]))
    return (rot, trans)


def rots_mul_rots(x, y):
    r"""
    Get result of rotation matrix x multiply rotation matrix y.

    .. math::
        \begin{split}
        &xx = xx1*xx2 + xy1*yx2 + xz1*zx2 \\
        &xy = xx1*xy2 + xy1*yy2 + xz1*zy2 \\
        &xz = xx1*xz2 + xy1*yz2 + xz1*zz2 \\
        &yx = yx1*xx2 + yy1*yx2 + yz1*zx2 \\
        &yy = yx1*xy2 + yy1*yy2 + yz1*zy2 \\
        &yz = yx1*xz2 + yy1*yz2 + yz1*zz2 \\
        &zx = zx1*xx2 + zy1*yx2 + zz1*zx2 \\
        &zy = zx1*xy2 + zy1*yy2 + zz1*zy2 \\
        &zz = zx1*xz2 + zy1*yz2 + zz1*zz2 \\
        \end{split}

    Args:
        x(tuple):   rots x, :math:`(xx1, xy1, xz1, yx1, yy1, yz1, zx1, zy1, zz1)`.
        y(tuple):   rots y, :math:`(xx2, xy2, xz2, yx2, yy2, yz2, zx2, zy2, zz2)`.

    Returns:
        tuple, the result of rots x multiplying rots y. The result is :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindsponge.common.geometry import rots_mul_rots
        >>> rtos_0 = (1, 1, 1, 1, 1, 1, 1)
        >>> rtos_1 = (1, 1, 1, 1, 1, 1, 1)
        >>> result = rots_mul_rots(rots_0, rots_1)
        >>> print(output)
        (3, 3, 3, 3, 3, 3, 3, 3, 3)
    """
    vecs0 = rots_mul_vecs(x, (y[0], y[3], y[6]))
    vecs1 = rots_mul_vecs(x, (y[1], y[4], y[7]))
    vecs2 = rots_mul_vecs(x, (y[2], y[5], y[8]))
    rots = (vecs0[0], vecs1[0], vecs2[0], vecs0[1], vecs1[1], vecs2[1], vecs0[2], vecs1[2], vecs2[2])
    return rots


def vecs_from_tensor(inputs):
    """
    Get vectors from the last axis of input tensor.

    Args:
        inputs(Tensor): Atom position information. Shape is :math:`(..., 3)`.

    Returns:
        tuple :math:`(x, y, z)` with three tensors,
        including the coordinate information of x, y and z.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindsponge.common.geometry import vecs_from_tensor
        >>> input_0 = Tensor(np.ones((4, 256, 3)), ms.float32)
        >>> output = vecs_from_tensor(input_0)
        >>> print(len(output), output[0].shape)
        3, (4,256)
    """
    num_components = inputs.shape[-1]
    assert num_components == 3
    return (inputs[..., 0], inputs[..., 1], inputs[..., 2])


def vecs_to_tensor(v):
    """
    Converts 'v' to tensor with last dim shape 3, inverse of 'vecs_from_tensor'.

    Args:
        v(tuple):   Input tuple v :math:`(x, y, z)` with three tensors, including
                    the coordinate information of x, y and z.

    Returns:
        tensor, concat the tensor in last dims, shape :math:`(..., 3)` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindsponge.common.geometry import vecs_to_tensor
        >>> input_0 = Tensor(np.ones((4, 256)), ms.float32)
        >>> input_1 = Tensor(np.ones((4, 256)), ms.float32)
        >>> input_2 = Tensor(np.ones((4, 256)), ms.float32)
        >>> inputs = (input_0, input_1, input_2)
        >>> output = vecs_to_tensor(inputs)
        >>> print(output.shape)
        (4, 256, 3)
    """
    return mnp.stack([v[0], v[1], v[2]], axis=-1)


def make_transform_from_reference(point_a, point_b, point_c):
    r"""
    Using GramSchmidt process to construct rotation and translation from given points.

    Calculate the rotation matrix and translation meets

    a) point_b is the original point.

    b) point_c is on the x_axis.

    c) the plane a-b-c is on the x-y plane.

    .. math::
        \begin{split}
        &\vec v_1 = \vec x_3 - \vec x_2 \\
        &\vec v_2 = \vec x_1 - \vec x_2 \\
        &\vec e_1 = \vec v_1 / ||\vec v_1|| \\
        &\vec u_2 = \vec v_2 - \vec  e_1(\vec e_1^T\vec v_2) \\
        &\vec e_2 = \vec u_2 / ||\vec u_2|| \\
        &\vec e_3 = \vec e_1 \times \vec e_2 \\
        &rotation = (\vec e_1, \vec e_2, \vec e_3) \\
        &translation = (\vec x_2) \\
        \end{split}

    Args:
        point_a(float, tensor) -> (tensor): Spatial location information of atom 'N',
                                            shape is :math:`[..., N_{res}, 3]` .
        point_b(float, tensor) -> (tensor): Spatial location information of atom 'CA',
                                            shape is :math:`[..., N_{res}, 3]` .
        point_c(float, tensor) -> (tensor): Spatial location information of atom 'C',
                                            shape is :math:`[..., N_{res}, 3]` .

    Returns:
        - Tuple, rots :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` ,
          the shape of every element is :math:`(..., N_{res})` .
        - Tuple, trans :math:`(x, y, z)` , the shape of every element is :math:`(..., N_{res})` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindsponge.common.geometry import make_transform_from_reference
        >>> input_0 = Tensor(np.ones((4, 256, 3)), ms.float32)
        >>> input_1 = Tensor(np.ones((4, 256, 3)), ms.float32)
        >>> input_2 = Tensor(np.ones((4, 256, 3)), ms.float32)
        >>> rots, trans = make_transform_from_reference(input_0, input_1, input_2)
        >>> print(len(rots), rots[0].shape, len(trans), trans[0].shape)
        9, (4, 256), 3, (4, 256)
    """

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
    """
    Amortize and split the 3*3 rotation matrix corresponding to the last two axes of input Tensor
      to obtain each component of the rotation matrix, inverse of 'rots_to_tensor'.

    Args:
        rots(Tensor):       Represent the rotation matrix, shape is :math:`(..., 3, 3)` .
        use_numpy(bool):    Whether to use numpy to calculate. Default: False.

    Returns:
        Tuple, rots represented by vectors, rots is :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindsponge.common.geometry import rots_from_tensor
        >>> input_0 = Tensor(np.ones((256, 3, 3)), ms.float32)
        >>> output = rots_from_tensor(input_0)
        >>> print(len(output), output[0].shape)
        9, (256,)
    """
    if use_numpy:
        rots = np.reshape(rots, rots.shape[:-2] + (9,))
    else:
        rots = P.Reshape()(rots, P.Shape()(rots)[:-2] + (9,))
    rotation = (rots[..., 0], rots[..., 1], rots[..., 2],
                rots[..., 3], rots[..., 4], rots[..., 5],
                rots[..., 6], rots[..., 7], rots[..., 8])
    return rotation


def rots_to_tensor(rots, use_numpy=False):
    """
    Translate rots represented by vectors to tensor, inverse of 'rots_from_tensor'.

    Args:
        rots(Tuple):        Rots represented by vectors, shape is :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` .
        use_numpy(bool):    Whether to use numpy to calculate. Default: False.

    Returns:
        Tensor, concat the tensor in last dims, shape :math:`(N_{res}, 3, 3)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindsponge.common.geometry import rots_to_tensor
        >>> inputs = [Tensor(np.ones((256,)), ms.float32) for i in range(9)]
        >>> output = rots_to_tensor(inputs)
        >>> print(output.shape)
        (256, 3, 3)
    """
    assert len(rots) == 9
    if use_numpy:
        rots = np.stack(rots, axis=-1)
        rots = np.reshape(rots, rots.shape[:-1] + (3, 3))
    else:
        rots = mnp.stack(rots, axis=-1)
        rots = mnp.reshape(rots, rots.shape[:-1] + (3, 3))
    return rots


def quat_affine(quaternion, translation, rotation=None, normalize=True, unstack_inputs=False, use_numpy=False):
    """
    Create quat affine representations based on rots and trans.

    Args:
        quaternion(tensor):     Shape is :math:`(N_{res}, 4)`.
        translation(tensor):    Shape is :math:`(N_{res}, 3)`.
        rotation(tensor):       Rots, shape is :math:`(N_{res}, 9)`. Default: None.
        normalize(bool):        Whether to use normalization. Default: True.
        unstack_inputs(bool):   Whether input is vector(True) of Tensor(False). Default: False.
        use_numpy(bool):        Whether to use numpy. Default: False.

    Returns:
        result after quat affine.

        - quaternion, tensor, shape is :math:`(N_{res}, 4)` .
        - rotation, tuple, :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`,
          shape of every element is :math:`(N_{res},)` .
        - translation, tensor, shape is :math:`(N_{res}, 3)` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindsponge.common.geometry import quat_affine
        >>> input_0 = Tensor(np.ones((256, 4)), ms.float32)
        >>> input_1 = Tensor(np.ones((256, 3)), ms.float32)
        >>> qua, rot, trans = quat_affine(input_0, input_1)
        >>> print(qua.shape, len(rot), rot[0].shape, trans.shape)
        (256, 4), 9, (256,), (256, 3)
    """
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
    r"""
    Convert a normalized quaternion to a rotation matrix.

    .. math::
        \begin{split}
        &xx = 1 - 2 * y * y - 2 * z * z \\
        &xy = 2 * x * y + 2 * w * z \\
        &xz = 2 * x * z - 2 * w * y \\
        &yx = 2 * x * y - 2 * w * z \\
        &yy = 1 - 2 * x * x - 2 * z * z \\
        &yz = 2 * z * y + 2 * w * x \\
        &zx = 2 * x * z + 2 * w * y \\
        &zy = 2 * y * z - 2 * w * x \\
        &zz = 1 - 2 * x * x - 2 * y * y \\
        \end{split}

    Args:
        normalized_quat (tensor): normalized quaternion, shape :math:`(N_{res}, 4)`.
        use_numpy (bool): use numpy or not, Default: "False".

    Returns:
        tuple, rotation :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`, every element shape :math:`(N_{res}, )`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindsponge.common.geometry import quat_to_rot
        >>> input_0 = Tensor(np.ones((256, 4)), ms.float32)
        >>> output = quat_to_rot(input_0)
        >>> print(len(output), output[0].shape)
        9, (256,)
    """
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
    """
    Initialize quaternion, rotation, translation of affine.

    Args:
        num_residues(int):  Number of residues.
        use_numpy(bool):    Whether to use numpy. Default: False.

    Returns:
        result after quat affine.
        - quaternion, tensor, shape is :math:`(N_{res}, 4)` .
        - rotation, tuple, :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`, shape of every element is :math:`(N_{res}, )` .
        - translation, tuple, :math:`(x, y, z)` shape of every element tensor is :math:`(N_{res}, )` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindsponge.common.geometry import initial_affine
        >>> output = initial_affine(256)
        >>> print(len(output), output[0].shape, len(output[1]), len(output[1][0]), len(output[2]), len(output[2][0]))
        >>> print(output[0])
        >>> print(output[1])
        >>> print(output[2])
        3, (1, 4), 9, 1, 3, 1
        [[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]
        (1, 0, 0, 0, 1, 0, 0, 0, 1)
        ([0.00000000e+00], [0.00000000e+00], [0.00000000e+00])
    """
    if use_numpy:
        quaternion = np.tile(np.reshape(np.asarray([1., 0., 0., 0.]), [1, 4]), [num_residues, 1])
        translation = np.zeros([num_residues, 3])
    else:
        quaternion = mnp.tile(mnp.reshape(mnp.asarray([1., 0., 0., 0.]), [1, 4]), [num_residues, 1])
        translation = mnp.zeros([num_residues, 3])
    return quat_affine(quaternion, translation, unstack_inputs=True, use_numpy=use_numpy)


def vecs_expand_dims(v, axis):
    r"""
    Add an extra dimension to the input `v` at the given axis.

    Args:
        v(Tuple):   Input vector. Length is 3, :math:`(xx, xy, xz)` .
        axis(int):  Specifies the dimension index at which to expand the shape of `v`. Only constant value is allowed.

    Returns:
        Tuple, if the axis is 0, and the shape of :math:`xx` is :math:`(..., X_R)`, where X_R is any number.
          The expanded shape is :math:`(1, ..., X_R)`. If the axis is other value, then expand in the other
          direction. And return expanded :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindsponge.common.geometry import vecs_expand_dims
        >>> from mindspore.common import Tensor
        >>> from mindspore import dtype as mstype
        >>> v = (1, 2, 3)
        >>> axis = 0
        >>> output= vecs_expand_dims(v, axis)
        >>> print(output)
        (Tensor(shape=[1], dtype=Int64, value=[1]),Tensor(shape=[1], dtype=Int64, value=[2]),
         Tensor(shape=[1], dtype=Int64, value=[3]))
    """
    v = (P.ExpandDims()(v[0], axis), P.ExpandDims()(v[1], axis), P.ExpandDims()(v[2], axis))
    return v


def rots_expand_dims(rots, axis):
    """
    Adds an additional dimension to `rots` at the given axis.

    Args:
        rots (Tuple):   The rotation matrix is :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`,
                        and xx and xy have the same shape
        axis (Int):     Specifies the dimension index at which to expand the shape of v.
                        Only constant value is allowed.

    Returns:
        Tuple, rots. If the value of axis is 0, and the shape of xx is :math:`(..., X_R)`,
          where X_R is any number, and the expanded shape is :math:`(1, ..., X_R)`.
          Return expanded :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindsponge.common.geometry import rots_expand_dims
        >>> from mindspore.common import Tensor
        >>> from mindspore import dtype as mstype
        >>> rots = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        >>> axis = 0
        >>> rots_expand_dims(rots, axis)
        >>> print(output)
        (Tensor(shape=[1], dtype=Int64, value=[1]), Tensor(shape=[1], dtype=Int64, value=[2]),
        Tensor(shape=[1], dtype=Int64, value=[3]), Tensor(shape=[1], dtype=Int64, value=[4]),
        Tensor(shape=[1], dtype=Int64, value=[5]), Tensor(shape=[1], dtype=Int64, value=[6]),
        Tensor(shape=[1], dtype=Int64, value=[7]), Tensor(shape=[1], dtype=Int64, value=[8]),
        Tensor(shape=[1], dtype=Int64, value=[9]))
    """
    rots = (P.ExpandDims()(rots[0], axis), P.ExpandDims()(rots[1], axis), P.ExpandDims()(rots[2], axis),
            P.ExpandDims()(rots[3], axis), P.ExpandDims()(rots[4], axis), P.ExpandDims()(rots[5], axis),
            P.ExpandDims()(rots[6], axis), P.ExpandDims()(rots[7], axis), P.ExpandDims()(rots[8], axis))
    return rots


def invert_point(transformed_point, rotation, translation, extra_dims=0, stack=False, use_numpy=False):
    r"""
    The inverse transformation of a rigid body group transformation with respect to a point coordinate,
    that is, the inverse transformation of apply to point Make rotational translation changes on coordinates
    with the transpose of the rotation
    matrix :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)` and the translation vector :math:`(x, y, z)` translation.

    First, the initial coordinates are translated, and then the transpose of the rotation matrix is multiplied
    by rot_point to get the final coordinates.

    .. math::
        \begin{split}
        &rot\_point = transformed\_point - translation \\
        &result = rotation^T * rot\_point \\
        \end{split}

    The specific procedures of vector subtraction, transpose and multiplication can be referred to the
    api of vecs_sub, invert_rots, rots_mul_vecs etc.

    Args:
        transformed_point (Tuple):  The initial coordinates of the input have shape :math:`(x, y, z)`,
                                    where x, y and z are Tensor and have the same shape.
        rotation (Tuple):           The rotation matrix. shape is :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`,
                                    and xx and xy have the same shape.
        translation (Tuple):        The translation vector shape is :math:`(x, y, z)`,
                                    where x, y and z are Tensor and have the same shape.
        extra_dims (int):           Control whether to expand dims. Default: 0.
        stack (bool):               Control whether to transform to tuple. Default: False.
        use_numpy(bool):            Control whether to use numpy. Default: False.

    Returns:
        Tuple, the transformed coordinate of invert point.Length is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindsponge.common.geometry import invert_point
        >>> from mindspore.common import Tensor
        >>> from mindspore import dtype as mstype
        >>> transformed_point = (1, 2, 3)
        >>> rotation = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        >>> translation = (1, 0.5, -1)
        >>> output= invert_point(transformed_point, rotation, translation)
        >>> print(output)
        (Tensor(shape=[], dtype=Float32, value = 34), Tensor(shape=[], dtype=Float32, value = 39.5),
         Tensor(shape=[], dtype=Float32, value = 45))
    """
    if stack:
        rotation = rots_from_tensor(rotation, use_numpy)
        translation = vecs_from_tensor(translation)
    for _ in range(extra_dims):
        rotation = rots_expand_dims(rotation, -1)
        translation = vecs_expand_dims(translation, -1)
    rot_point = vecs_sub(transformed_point, translation)
    return rots_mul_vecs(invert_rots(rotation), rot_point)


def quat_multiply_by_vec(quat, vec):
    r"""
    Multiply a quaternion by a pure-vector quaternion.

    .. math::
        \begin{split}
        &temp =  QUAT\_MULTIPLY\_BY\_VEC * quat[..., :, None, None] * vec[..., None, :, None] \\
        &result = sum(temp,axis=(-3, -2)) \\
        \end{split}

    Args:
        quat (Tensor):  Quaternion.Tensor of shape :math:`(..., 4)`.
        vec (Tensor):   A pure-vector quaternion, :math:`(b, c, d)` not normalized quaternion.
                        Quaternion can be expressed as :math:`(1, b, c, d)`.

    Returns:
        Tensor, the product of a quaternion with a pure vector quaternion.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.common.geometry import quat_multiply_by_vec
        >>> from mindspore.common import Tensor
        >>> from mindspore import dtype as mstype
        >>> np.random.seed(1)
        >>> quat = Tensor(np.random.rand(4),dtype=mstype.float32)
        >>> vec = Tensor(np.random.rand(3),dtype=mstype.float32)
        >>> out = quat_multiply_by_vec(quat, vec)
        >>> print(out)
        [-0.16203496,  0.03330477, -0.05129148,  0.14417158]
    """

    return mnp.sum(QUAT_MULTIPLY_BY_VEC * quat[..., :, None, None] * vec[..., None, :, None],
                   axis=(-3, -2))


def pre_compose(quaternion, rotation, translation, update):
    r"""
    Return a new QuatAffine which applies the transformation update first.

    The process of obtaining the updated translation vector and rotation matrix is as follows:

    .. math::
        \begin{split}
        &update = (xx, xy, xz, yx, yy, yz) \\
        &vector\_quaternion\_update = (xx, xy, xz) \\
        &x = (yx) \\
        &y = (yy) \\
        &z = (yz) \\
        &trans\_update = (x, y, z) \\
        &new\_quaternion = quaternion + vector\_quaternion\_update * quaternion \\
        &rotated\_trans\_update = rotation * trans\_update \\
        &new\_translation = translation + rotated\_trans\_update \\
        \end{split}

    vector_quaternion_update and quaternion are multiplied by the quat_multiply_by_vec function,
    Affine transformation is performed using the generated new_quaternion and new_translation.
    The process of affine transformation is referred to the quat_affine api.

    Args:
        quaternion (Tensor):    The initial quaternion to be updated, shape :math:`[(..., 4)]`.
        rotation (Tuple):       Rotation matrix, :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`,
                                and xx and xy are Tensor and have the same shape.
        translation (Tuple):    Translation vector :math:`(x, y, z)`,
                                where x, y and z are Tensor and have the same shape.
        update (Tensor):        The update-assisted matrix has shape :math:`[(..., 6)]`.
                                3-vector of x, y, and z such that the quaternion
                                update is :math:`(1, x, y, z)` and zero for the 3-vector is the identity
                                quaternion. 3-vector for translation concatenated.

    Returns:
        - Tensor, new quaternion.The updated Tensor tuple has shape :math:`[(..., 4)]`.
        - Tuple, the updated rotation matrix :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`,
          and xx and xy are Tensor and have the same shape.
        - Tuple, the updated translation vector :math:`(x, y, z)`,
          where x, y and z are Tensor and have the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.common.geometry import pre_compose
        >>> from mindspore.common import Tensor
        >>> from mindspore import dtype as mstype
        >>> np.random.seed(1)
        >>> quaternion = Tensor(np.random.rand(4),dtype=mstype.float32)
        >>> update = Tensor(np.random.rand(6),dtype=mstype.float32)
        >>> rotation = Tensor(np.random.rand(9),dtype=mstype.float32)
        >>> translation = Tensor(np.random.rand(3),dtype=mstype.float32)
        >>> quaternion, rotation, translation = pre_compose(quaternion,rotation,translation,update)
        >>> print(quaternion)
        [ 0.27905196  0.82475466 -0.05600705  0.48864394]
        >>> print(rotation)
        (Tensor(shape=[], dtype=Float32, value= 0.516181), Tensor(shape=[], dtype=Float32, value= -0.365098),
        Tensor(shape=[], dtype=Float32, value= 0.774765), Tensor(shape=[], dtype=Float32, value= 0.18033),
        Tensor(shape=[], dtype=Float32, value= -0.837986), Tensor(shape=[], dtype=Float32, value= -0.515034),
        Tensor(shape=[], dtype=Float32, value= 0.837281), Tensor(shape=[], dtype=Float32, value= 0.405564),
        Tensor(shape=[], dtype=Float32, value= -0.366714))
        >>> print(translation)
        (Tensor(shape=[], dtype=Float32, value= 0.724994), Tensor(shape=[], dtype=Float32, value= 1.47631),
        Tensor(shape=[], dtype=Float32, value= 1.40978))
    """

    vector_quaternion_update, x, y, z = mnp.split(update, [3, 4, 5], axis=-1)
    trans_update = [mnp.squeeze(x, axis=-1), mnp.squeeze(y, axis=-1), mnp.squeeze(z, axis=-1)]
    new_quaternion = (quaternion + quat_multiply_by_vec(quaternion, vector_quaternion_update))
    rotated_trans_update = rots_mul_vecs(rotation, trans_update)
    new_translation = vecs_add(translation, rotated_trans_update)
    return quat_affine(new_quaternion, new_translation)


def quaternion_to_tensor(quaternion, translation):
    r"""
    Change quaternion to tensor.

    .. math::
        \begin{split}
        &quaternion = [(x_1, y_1, z_1, m_1)] \\
        &translation = [(x_2, y_2, z_2)] \\
        &result = [(x_1, y_1, z_1, m_1, x_2, y_2, z_2)] \\
        \end{split}

    Args:
        quaternion (Tensor):    Inputs quaternion. Tensor of shape :math:`(..., 4)`.
        translation (Tensor):    Inputs translation. Tensor of shape :math:`(..., 3)`

    Returns:
        Tensor, The result of the concatenation between translation and translation. Tensor of shape :math:`(..., 7)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.common.geometry import quaternion_to_tensor
        >>> from mindspore.common import Tensor
        >>> from mindspore import dtype as mstype
        >>> np.random.seed(1)
        >>> quaternion = Tensor(np.random.rand(4),dtype=mstype.float32)
        >>> translation = Tensor(np.random.rand(3),dtype=mstype.float32)
        >>> out = quaternion_to_tensor(quaternion, translation)
        >>> print(out)
        [0.6631489  0.44137922 0.97213906 0.7425225  0.3549025  0.6535310.5426164 ]
    """
    translation = (P.ExpandDims()(translation[0], -1), P.ExpandDims()(translation[1], -1),
                   P.ExpandDims()(translation[2], -1),)
    return mnp.concatenate((quaternion,) + translation, axis=-1)


def quaternion_from_tensor(tensor, normalize=False):
    r"""
    Take the input 'tensor' :math:`[(xx, xy, xz, yx, yy, yz, zz)]` to get the new
    'quaternion', 'rotation', 'translation'.

    .. math::
        \begin{split}
        &tensor = [(xx, xy, xz, yx, yy, yz, zz)] \\
        &quaternion = (xx, xy, xz, yx) \\
        &translation = (yy, yz, zz) \\
        \end{split}

    Affine transformation is performed using the generated quaternion and translation.
    The process of affine transformation is referred to the quat_affine api.

    Args:
        tensor(Tensor):     An initial Tensor :math:`[(xx, xy, xz, yx, yy, yz, zz)]` .
                            :math:`[(xx, xy, xz, yx)]` is the same with `quaternion`.
                            :math:`(yy, yz, zz)` is the same with `translation`.
        normalize(bool):    Control whether to find the norm during quat_affine. Default: False.

    Returns:
        - Tensor, new quaternion.Tensor of shape :math:`(..., 4)` .
        - Tuple, new rotation, :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`,
          and xx and xy are Tensor and have the same shape.
        - Tuple, translation vector :math:`[(x, y, z)]`, where x, y and z are Tensor and have the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.common.geometry import quaternion_from_tensor
        >>> from mindspore.common import Tensor
        >>> tensor = Tensor(np.random.rand(7),dtype=mstype.float32)
        >>> quaternion, rotation, translation = quaternion_from_tensor(tensor)
        >>> print(quaternion)
        [4.17021990e-01,  7.20324516e-01,  1.14374816e-04,  3.02332580e-01]
        >>> print(rotation)
        (Tensor(shape=[], dtype=Float32, value= 0.60137), Tensor(shape=[], dtype=Float32, value= -0.251994),
        Tensor(shape=[], dtype=Float32, value= 0.435651), Tensor(shape=[], dtype=Float32, value= 0.252323),
        Tensor(shape=[], dtype=Float32, value= -0.436365), Tensor(shape=[], dtype=Float32, value= -0.600713),
        Tensor(shape=[], dtype=Float32, value= 0.43546), Tensor(shape=[], dtype=Float32, value= 0.600851),
        Tensor(shape=[], dtype=Float32, value= -0.253555))
        >>> print(translation)
        (Tensor(shape=[], dtype=Float32, value= 0.146756),Tensor(shape=[], dtype=Float32, value= 0.0923386),
        Tensor(shape=[], dtype=Float32, value= 0.18626))
    """
    quaternion, tx, ty, tz = mnp.split(tensor, [4, 5, 6], axis=-1)
    translation = (P.Squeeze()(tx), P.Squeeze()(ty), P.Squeeze()(tz))
    return quat_affine(quaternion, translation, normalize=normalize)


def apply_to_point(rotation, translation, point, extra_dims=0):
    r"""
    Rotate and translate the input coordinates.

    .. math::
        \begin{split}
        &rot_point = rotation \cdot point \\
        &result = rot_point + translation \\
        \end{split}

    For specific multiplication and addition procedures, refer to the rots_mul_vecs and vecs_add apis.

    Args:
        rotation(Tuple):    The rotation matrix :math:`(xx, xy, xz, yx, yy, yz, zx, zy, zz)`,
                            and :math:`xx, xy` are Tensor and have the same shape.
        translation(Tuple): Translation vector :math:`[(x, y, z)]`,
                            where :math:`x, y, z` are Tensor and have the same shape.
        point(Tensor):      Initial coordinate values :math:`[(x, y, z)]`,
                            where :math:`x, y, z` are Tensor and have the same shape.
        extra_dims(int):    Control whether to expand dims. default:0.

    Returns:
        Tuple, the result of the coordinate transformation. Length is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.common.geometry import apply_to_point
        >>> from mindspore.common import Tensor
        >>> from mindspore import dtype as mstype
        >>> np.random.seed(1)
        >>> rotation = []
        >>> for i in range(9):
        ...     rotation.append(Tensor(np.random.rand(4),dtype=mstype.float32))
        >>> translation = []
        >>> for i in range(3):
        ...     translation.append(Tensor(np.random.rand(4),dtype=mstype.float32))
        >>> point = []
        >>> for i in range(3):
        ...     point.append(Tensor(np.random.rand(4),dtype=mstype.float32))
        >>> out = apply_to_point(rotation, translation, point)
        >>> print(out)
        (Tensor(shape=[4], dtype=Float32, value= [ 1.02389336e+00,  1.12493467e+00,  2.54357845e-01,  1.25249946e+00]),
        Tensor(shape=[4], dtype=Float32, value= [ 9.84841168e-01,  5.20081401e-01,  6.43978953e-01,  6.15328550e-01]),
        Tensor(shape=[4], dtype=Float32, value= [ 8.62860143e-01,  9.11733627e-01,  1.09284782e+00,  1.44202101e+00]))
    """
    for _ in range(extra_dims):
        rotation = rots_expand_dims(rotation, -1)
        translation = vecs_expand_dims(translation, -1)
    rot_point = rots_mul_vecs(rotation, point)
    result = vecs_add(rot_point, translation)
    return result
