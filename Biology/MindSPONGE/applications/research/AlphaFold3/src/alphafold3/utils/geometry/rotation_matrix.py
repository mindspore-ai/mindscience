# Copyright 2024 DeepMind Technologies Limited
# Copyright (C) 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications by Huawei Technologies Co., Ltd: Adapt to run by MindSpore on Ascend

"""Rot3Array Matrix Class."""

import dataclasses
from typing import Any, Final
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops, mint
from mindspore import Tensor
from alphafold3.utils.geometry import struct_of_array, utils, vector

COMPONENTS: Final[tuple[str, ...]] = (
    *('xx', 'xy', 'xz'),
    *('yx', 'yy', 'yz'),
    *('zx', 'zy', 'zz'),
)
VERSION: Final[str] = '0.1'


def make_matrix_svd_factors() -> Tensor:
    """Generates factors for converting 3x3 matrix to symmetric 4x4 matrix."""
    factors = mnp.zeros((16, 9), dtype=ms.float32)

    indices = [(0, [0, 4, 8]), ([1, 4], 5), ([1, 4], 7), ([2, 8], 6), ([2, 8], 2),
               ([3, 12], 1), ([3, 12], 3), (5, 0), (5, [4, 8]),
               ([6, 9], 1), ([6, 9], 3), ([7, 13], 2), ([7, 13], 6),
               (10, 4), (10, [0, 8]), ([11, 14], 5), ([11, 14], 7), (15, 8), (15, [0, 4])]

    values = [[1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0],
              [1.0, -1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
              [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
              [1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]

    for idx, val in zip(indices, values):
        if isinstance(idx[1], list):
            for i in idx[1]:
                factors[idx[0], i] = val[i % len(val)]
        else:
            factors[idx[0], idx[1]] = val[0]

    return factors


def largest_evec(m):
    _, eigvecs = np.linalg.eigh(m.asnumpy())

    return Tensor(eigvecs[..., -1])


MATRIX_SVD_QUAT_FACTORS = make_matrix_svd_factors()


@struct_of_array.StructOfArray(same_dtype=True)
class Rot3Array:
    """Rot3Array Matrix in 3 dimensional Space implemented as struct of arrays."""

    xx: Tensor = dataclasses.field(metadata={'dtype': ms.float32})
    xy: Tensor
    xz: Tensor
    yx: Tensor
    yy: Tensor
    yz: Tensor
    zx: Tensor
    zy: Tensor
    zz: Tensor

    __array_ufunc__ = None

    def inverse(self):
        """Returns inverse of Rot3Array."""
        return Rot3Array(
            *(self.xx, self.yx, self.zx),
            *(self.xy, self.yy, self.zy),
            *(self.xz, self.yz, self.zz),
        )

    def apply_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
        """Applies Rot3Array to point."""
        x = self.xx * point.x + self.xy * point.y + self.xz * point.z
        y = self.yx * point.x + self.yy * point.y + self.yz * point.z
        z = self.zx * point.x + self.zy * point.y + self.zz * point.z
        return vector.Vec3Array(x, y, z)

    def apply_inverse_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
        """Applies inverse Rot3Array to point."""
        return self.inverse().apply_to_point(point)

    def __matmul__(self, other):
        """Composes two Rot3Arrays."""
        c0 = self.apply_to_point(
            vector.Vec3Array(other.xx, other.yx, other.zx))
        c1 = self.apply_to_point(
            vector.Vec3Array(other.xy, other.yy, other.zy))
        c2 = self.apply_to_point(
            vector.Vec3Array(other.xz, other.yz, other.zz))
        return Rot3Array(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z)

    @classmethod
    def identity(cls, shape: Any, dtype: ms.dtype = ms.float32):
        """Returns identity of given shape."""
        ones = mint.ones(shape, dtype=dtype)
        zeros = mint.zeros(shape, dtype=dtype)

        temp = cls(ones, zeros, zeros, zeros, ones, zeros, zeros, zeros, ones)
        return temp

    @classmethod
    def from_two_vectors(cls, e0: vector.Vec3Array, e1: vector.Vec3Array):
        """Construct Rot3Array from two Vectors.

        Rot3Array is constructed such that in the corresponding frame 'e0' lies on
        the positive x-Axis and 'e1' lies in the xy plane with positive sign of y.

        Args:
          e0: Vector
          e1: Vector

        Returns:
          Rot3Array
        """
        # Normalize the unit vector for the x-axis, e0.
        e0 = e0.normalized()
        # Make e1 perpendicular to e0.
        c = e1.dot(e0)
        e1 = (e1 - e0 * c).normalized()
        # Compute e2 as cross product of e0 and e1.
        e2 = e0.cross(e1)
        return cls(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z, e2.z)

    @classmethod
    def from_array(cls, array: Tensor):
        """Construct Rot3Array Matrix from array of shape [..., 3, 3]."""
        unstacked = utils.unstack(array, axis=-2)
        unstacked = sum([utils.unstack(x, axis=-1) for x in unstacked], [])
        return cls(*unstacked)

    def to_array(self) -> Tensor:
        """Convert Rot3Array to array of shape [..., 3, 3]."""
        return ops.stack(
            [
                ops.stack([self.xx, self.xy, self.xz], axis=-1),
                ops.stack([self.yx, self.yy, self.yz], axis=-1),
                ops.stack([self.zx, self.zy, self.zz], axis=-1),
            ],
            axis=-2,
        )

    @classmethod
    def from_quaternion(
            cls,
            w: Tensor,
            x: Tensor,
            y: Tensor,
            z: Tensor,
            normalize: bool = True,
            epsilon: float = 1e-6,
    ):
        """Construct Rot3Array from components of quaternion."""
        if normalize:
            inv_norm = ops.rsqrt(ops.maximum(
                epsilon, w**2 + x**2 + y**2 + z**2))
            w *= inv_norm
            x *= inv_norm
            y *= inv_norm
            z *= inv_norm
        xx = 1 - 2 * (y**2 + z**2)
        xy = 2 * (x * y - w * z)
        xz = 2 * (x * z + w * y)
        yx = 2 * (x * y + w * z)
        yy = 1 - 2 * (x**2 + z**2)
        yz = 2 * (y * z - w * x)
        zx = 2 * (x * z - w * y)
        zy = 2 * (y * z + w * x)
        zz = 1 - 2 * (x**2 + y**2)
        return cls(xx, xy, xz, yx, yy, yz, zx, zy, zz)

    @classmethod
    def from_svd(cls, mat: Tensor, use_quat_formula: bool = True):
        """Constructs Rot3Array from arbitrary array of shape [3 * 3] using SVD.

        The case when 'use_quat_formula' is False rephrases the problem of
        projecting the matrix to a rotation matrix as a problem of finding the
        largest eigenvector of a certain 4x4 matrix. This has the advantage of
        having fewer numerical issues.
        This approach follows:
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.65.971&rep=rep1&type=pdf
        In the other case we construct it via svd following
        https://arxiv.org/pdf/2006.14616.pdf
        In that case [∂L/∂M] is large if the two smallest singular values are close
        to each other, or if they are close to 0.

        Args:
          mat: Array of shape [..., 3 * 3]
          use_quat_formula: Whether to construct matrix via 4x4 eigenvalue problem.

        Returns:
          Rot3Array of shape [...]
        """
        assert mat.shape[-1] == 9
        if use_quat_formula:
            symmetric_4by4 = ops.einsum(
                'ji, ...i -> ...j',
                MATRIX_SVD_QUAT_FACTORS,
                mat,
            )
            symmetric_4by4 = ops.reshape(
                symmetric_4by4, mat.shape[:-1] + (4, 4))
            largest_eigvec = largest_evec(symmetric_4by4)
            return cls.from_quaternion(
                *utils.unstack(largest_eigvec, axis=-1)
            ).inverse()

        mat = ops.reshape(mat, mat.shape[:-1] + (3, 3))
        u, _, v_t = np.linalg.svd(mat.asnumpy(), full_matrices=False)
        u = Tensor(u)
        v_t = Tensor(v_t)
        det_uv_t = ops.det(ops.matmul(u, v_t))
        ones = ops.ones_like(det_uv_t)
        diag_array = ops.stack([ones, ones, det_uv_t], axis=-1)
        # This is equivalent to making diag_array into a diagonal array and matrix
        # multiplying
        diag_times_v_t = diag_array[..., None] * v_t
        out = ops.matmul(u, diag_times_v_t)
        return cls.from_array(out)

    @classmethod
    def random_uniform(cls, key, shape, dtype=ms.float32):
        """Samples uniform random Rot3Array according to Haar Measure."""
        stdnormal = ops.StandardNormal(seed=key)
        quat_array = stdnormal(shape + (4,)).astype(dtype)
        # quat_array = ops.StandardNormal()(shape=(tuple(shape) + (4,)), seed=key)
        quats = utils.unstack(quat_array)
        return cls.from_quaternion(*quats)

    def __getstate__(self):
        return (VERSION, [getattr(self, field) for field in COMPONENTS])

    def __setstate__(self, state):
        version, state = state
        del version
        for i, field in enumerate(COMPONENTS):
            object.__setattr__(self, field, state[i])
