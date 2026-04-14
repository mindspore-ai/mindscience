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

"""Vec3Array Class."""

import dataclasses
from typing import Final, TypeVar, TypeAlias

import mindspore as ms
from mindspore import ops, mint
from alphafold3.utils.geometry import struct_of_array

Self = TypeVar('Self', bound='Vec3Array')
Float = TypeAlias = float | ms.Tensor
VERSION: Final[str] = '0.1'


def tree_map(func, *trees):
    """
    Recursively applies a function to each leaf of the input trees.

    Args:
        func: A function to apply to each leaf.
        *trees: One or more tree structures (nested lists/tuples/dicts).

    Returns:
        A new tree with the same structure where `func` has been applied to each leaf.
    """
    if isinstance(trees[0], Vec3Array):
        return Vec3Array(
            x=tree_map(func, *(t.x for t in trees)),
            y=tree_map(func, *(t.y for t in trees)),
            z=tree_map(func, *(t.z for t in trees))
        )
    if isinstance(trees[0], dict):
        return {key: tree_map(func, *(t[key] for t in trees)) for key in trees[0]}
    if isinstance(trees[0], (list, tuple)):
        return type(trees[0])(tree_map(func, *args) for args in zip(*trees))
    return func(*trees)


@struct_of_array.StructOfArray(same_dtype=True)
class Vec3Array:
    """Vec3Array in 3 dimensional Space implemented as struct of arrays.
    This is done in order to improve performance and precision.
    """

    x: ms.Tensor = dataclasses.field(metadata={'dtype': ms.float32})
    y: ms.Tensor
    z: ms.Tensor

    def __post_init__(self):
        if hasattr(self.x, 'dtype'):
            if not self.x.dtype == self.y.dtype == self.z.dtype:
                raise ValueError(
                    f'Type mismatch: {self.x.dtype}, {self.y.dtype}, {self.z.dtype}'
                )
            if not self.x.shape == self.y.shape == self.z.shape:
                raise ValueError(
                    f'Shape mismatch: {self.x.shape}, {self.y.shape}, {self.z.shape}'
                )

    @property
    def shape(self):
        """Return the shape of the Vec3Array."""
        return self.x.shape

    def __add__(self, other: Self) -> Self:
        return tree_map(ops.add, self, other)

    def __sub__(self, other: Self) -> Self:
        return tree_map(ops.sub, self, other)

    def __mul__(self, other: Float | ms.Tensor) -> Self:
        if isinstance(other, float):
            return tree_map(lambda x: ops.mul(x, other), self)
        x = ops.mul(self.x, other)
        y = ops.mul(self.y, other)
        z = ops.mul(self.z, other)
        return Vec3Array(x, y, z)

    def __rmul__(self, other: Float | ms.Tensor) -> Self:
        if isinstance(other, float):
            return self * other
        x = ops.mul(self.x, other)
        y = ops.mul(self.y, other)
        z = ops.mul(self.z, other)
        return Vec3Array(x, y, z)

    def __truediv__(self, other: Float) -> Self:
        return tree_map(lambda x: ops.div(x, other), self)

    def __neg__(self) -> Self:
        return tree_map(lambda x: -x, self)

    def __pos__(self) -> Self:
        return tree_map(lambda x: x, self)

    def cross(self, other: Self) -> Self:
        """Compute cross product between 'self' and 'other'."""
        new_x = ops.sub(ops.mul(self.y, other.z), ops.mul(self.z, other.y))
        new_y = ops.sub(ops.mul(self.z, other.x), ops.mul(self.x, other.z))
        new_z = ops.sub(ops.mul(self.x, other.y), ops.mul(self.y, other.x))
        return Vec3Array(new_x, new_y, new_z)

    def dot(self, other: Self) -> ms.Tensor:
        """Compute dot product between 'self' and 'other'."""
        return ops.add(ops.add(ops.mul(self.x, other.x), ops.mul(self.y, other.y)), ops.mul(self.z, other.z))

    def norm(self, epsilon: float = 1e-6) -> ms.Tensor:
        """Compute Norm of Vec3Array, clipped to epsilon."""
        # To avoid NaN on the backward pass, we must use maximum before the sqrt
        norm2 = self.dot(self)
        if epsilon:
            norm2 = ops.maximum(norm2, epsilon**2)
        return ops.sqrt(norm2)

    def norm2(self) -> ms.Tensor:
        return self.dot(self)

    def normalized(self, epsilon: float = 1e-6) -> Self:
        """Return unit vector with optional clipping."""
        return self / self.norm(epsilon)

    @classmethod
    def zeros(cls, shape, dtype=ms.float32):
        """Return Vec3Array corresponding to zeros of given shape."""
        return cls(
            mint.zeros(shape, dtype=dtype),
            mint.zeros(shape, dtype=dtype),
            mint.zeros(shape, dtype=dtype),
        )

    def to_array(self) -> ms.Tensor:
        return ops.stack([self.x, self.y, self.z], axis=-1)

    @classmethod
    def from_array(cls, array):
        unstacked = ops.unstack(array, axis=-1)
        return cls(unstacked[0], unstacked[1], unstacked[2])

    def __getstate__(self):
        return (
            VERSION,
            [self.x.asnumpy(), self.y.asnumpy(), self.z.asnumpy()],
        )

    def __setstate__(self, state):
        version, state = state
        del version
        for i, letter in enumerate('xyz'):
            object.__setattr__(self, letter, ms.Tensor(state[i]))


def square_euclidean_distance(
        vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 1e-6
) -> Float:
    """Computes square of euclidean distance between 'vec1' and 'vec2'.

    Args:
      vec1: Vec3Array to compute  distance to
      vec2: Vec3Array to compute  distance from, should be broadcast compatible
        with 'vec1'
      epsilon: distance is clipped from below to be at least epsilon

    Returns:
      Array of square euclidean distances;
      shape will be result of broadcasting 'vec1' and 'vec2'
    """
    difference = vec1 - vec2
    distance = difference.dot(difference)
    if epsilon:
        distance = ops.maximum(distance, epsilon)
    return distance


def dot(vector1: Vec3Array, vector2: Vec3Array) -> Float:
    return vector1.dot(vector2)


def cross(vector1: Vec3Array, vector2: Vec3Array) -> Float:
    return vector1.cross(vector2)


def norm(vector: Vec3Array, epsilon: float = 1e-6) -> Float:
    return vector.norm(epsilon)


def normalized(vector: Vec3Array, epsilon: float = 1e-6) -> Vec3Array:
    return vector.normalized(epsilon)


def euclidean_distance(
        vec1: Vec3Array, vec2: Vec3Array, epsilon: float = 1e-6
) -> Float:
    """Computes euclidean distance between 'vec1' and 'vec2'.

    Args:
      vec1: Vec3Array to compute euclidean distance to
      vec2: Vec3Array to compute euclidean distance from, should be broadcast
        compatible with 'vec1'
      epsilon: distance is clipped from below to be at least epsilon

    Returns:
      Array of euclidean distances;
      shape will be result of broadcasting 'vec1' and 'vec2'
    """
    distance_sq = square_euclidean_distance(vec1, vec2, epsilon**2)
    distance = ops.sqrt(distance_sq)
    return distance


def dihedral_angle(
        a: Vec3Array, b: Vec3Array, c: Vec3Array, d: Vec3Array
) -> Float:
    """Computes torsion angle for a quadruple of points.

    For points (a, b, c, d), this is the angle between the planes defined by
    points (a, b, c) and (b, c, d). It is also known as the dihedral angle.

    Arguments:
      a: A Vec3Array of coordinates.
      b: A Vec3Array of coordinates.
      c: A Vec3Array of coordinates.
      d: A Vec3Array of coordinates.

    Returns:
      A tensor of angles in radians: [-pi, pi].
    """
    v1 = a - b
    v2 = b - c
    v3 = d - c

    c1 = v1.cross(v2)
    c2 = v3.cross(v2)
    c3 = c2.cross(c1)

    v2_mag = v2.norm()
    return ops.atan2(c3.dot(v2), v2_mag * c1.dot(c2))


def random_gaussian_vector(shape, key=None, dtype=ms.float32) -> Vec3Array:
    stdnormal = ops.StandardNormal(seed=key)
    vec_array = stdnormal(shape + (3,)).astype(dtype)
    return Vec3Array.from_array(vec_array)
