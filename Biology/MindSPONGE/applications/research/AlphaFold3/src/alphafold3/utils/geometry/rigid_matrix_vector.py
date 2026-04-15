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

"""Rigid3Array Transformations represented by a Matrix and a Vector."""

from typing import Any, Final, TypeAlias
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.ops import operations as P

from alphafold3.utils.geometry import rotation_matrix, struct_of_array, utils, vector

Float: TypeAlias = float | Tensor

VERSION: Final[str] = '0.1'


def _compute_covariance_matrix(
        row_values: vector.Vec3Array,
        col_values: vector.Vec3Array,
        weights: Tensor,
        epsilon=1e-6,
) -> Tensor:
    """Compute covariance matrix."""
    weights = mnp.asarray(weights)

    weights = mnp.broadcast_to(weights, row_values.shape)

    normalized_weights = weights / \
        (mnp.sum(weights, axis=-1, keepdims=True) + epsilon)

    def weighted_average(x):
        return mnp.sum(normalized_weights * x, axis=-1)

    out = [
        mnp.stack(
            (
                weighted_average(row_values.x * col_values.x),
                weighted_average(row_values.x * col_values.y),
                weighted_average(row_values.x * col_values.z),
            ),
            axis=-1,
        )
    ]

    out.append(
        mnp.stack(
            (
                weighted_average(row_values.y * col_values.x),
                weighted_average(row_values.y * col_values.y),
                weighted_average(row_values.y * col_values.z),
            ),
            axis=-1,
        )
    )

    out.append(
        mnp.stack(
            (
                weighted_average(row_values.z * col_values.x),
                weighted_average(row_values.z * col_values.y),
                weighted_average(row_values.z * col_values.z),
            ),
            axis=-1,
        )
    )

    return mnp.stack(out, axis=-2)


@struct_of_array.StructOfArray(same_dtype=True)
class Rigid3Array:
    """Rigid Transformation, i.e. element of special euclidean group."""

    rotation: rotation_matrix.Rot3Array
    translation: vector.Vec3Array

    def __matmul__(self, other: 'Rigid3Array') -> 'Rigid3Array':
        new_rotation = self.rotation @ other.rotation
        new_translation = self.apply_to_point(other.translation)
        return Rigid3Array(new_rotation, new_translation)

    def inverse(self) -> 'Rigid3Array':
        """Return Rigid3Array corresponding to inverse transform."""
        inv_rotation = self.rotation.inverse()
        inv_translation = inv_rotation.apply_to_point(-self.translation)
        return Rigid3Array(inv_rotation, inv_translation)

    def apply_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
        """Apply Rigid3Array transform to point."""
        return self.rotation.apply_to_point(point) + self.translation

    def apply_inverse_to_point(self, point: vector.Vec3Array) -> vector.Vec3Array:
        """Apply inverse Rigid3Array transform to point."""
        new_point = point - self.translation
        return self.rotation.apply_inverse_to_point(new_point)

    def compose_rotation(self, other_rotation: rotation_matrix.Rot3Array) -> 'Rigid3Array':
        rot = self.rotation @ other_rotation
        trans = P.BroadcastTo(rot.shape)(self.translation)
        return Rigid3Array(rot, trans)

    @classmethod
    def identity(cls, shape: Any, dtype: ms.dtype = ms.float32) -> 'Rigid3Array':
        """Return identity Rigid3Array of given shape."""

        return cls(
            rotation_matrix.Rot3Array.identity(shape, dtype=dtype),
            vector.Vec3Array.zeros(shape, dtype=dtype),
        )

    def scale_translation(self, factor: Float) -> 'Rigid3Array':
        """Scale translation in Rigid3Array by 'factor'."""
        return Rigid3Array(self.rotation, self.translation * factor)

    def to_array(self):
        rot_array = self.rotation.to_array()
        vec_array = self.translation.to_array()
        return mnp.concatenate([rot_array, vec_array[..., None]], axis=-1)

    @classmethod
    def from_array(cls, array):
        rot = rotation_matrix.Rot3Array.from_array(array[..., :3])
        vec = vector.Vec3Array.from_array(array[..., -1])
        return cls(rot, vec)

    @classmethod
    def from_array4x4(cls, array: Tensor) -> 'Rigid3Array':
        """Construct Rigid3Array from homogeneous 4x4 array."""
        if array.shape[-2:] != (4, 4):
            raise ValueError(f'array.shape({array.shape}) must be [..., 4, 4]')
        rotation = rotation_matrix.Rot3Array(
            *(array[..., 0, 0], array[..., 0, 1], array[..., 0, 2]),
            *(array[..., 1, 0], array[..., 1, 1], array[..., 1, 2]),
            *(array[..., 2, 0], array[..., 2, 1], array[..., 2, 2]),
        )
        translation = vector.Vec3Array(
            array[..., 0, 3], array[..., 1, 3], array[..., 2, 3]
        )
        return cls(rotation, translation)

    @classmethod
    def from_point_alignment(
            cls,
            points_to: vector.Vec3Array,
            points_from: vector.Vec3Array,
            weights: Float | None = None,
            epsilon: float = 1e-6,
    ) -> 'Rigid3Array':
        """Constructs Rigid3Array by finding transform aligning points."""
        if weights is None:
            weights = 1.0

        def compute_center(value):
            return utils.weighted_mean(value=value, weights=weights, axis=-1)

        points_to_center = P.Map()(compute_center, points_to)
        points_from_center = P.Map()(compute_center, points_from)
        centered_points_to = points_to - points_to_center[..., None]
        centered_points_from = points_from - points_from_center[..., None]
        cov_mat = _compute_covariance_matrix(
            centered_points_to,
            centered_points_from,
            weights=weights,
            epsilon=epsilon,
        )
        rots = rotation_matrix.Rot3Array.from_svd(
            mnp.reshape(cov_mat, cov_mat.shape[:-2] + (9,))
        )

        translations = points_to_center - \
            rots.apply_to_point(points_from_center)

        return cls(rots, translations)

    def __getstate__(self):
        return (VERSION, (self.rotation, self.translation))

    def __setstate__(self, state):
        version, (rot, trans) = state
        del version
        object.__setattr__(self, 'rotation', rot)
        object.__setattr__(self, 'translation', trans)
