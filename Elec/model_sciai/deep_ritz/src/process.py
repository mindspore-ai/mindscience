
# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Data processing"""
import math

import numpy as np


def sample_from_disk(r, n):
    """
    Sample points in a disk.
    Args:
        r: Radius;
        n: Number of samples.
    """
    array = np.random.rand(2 * n, 2) * 2 * r - r
    array = np.multiply(array.T, (np.linalg.norm(array, 2, axis=1) < r)).T
    array = array[~np.all(array == 0, axis=1)]

    if np.shape(array)[0] >= n:
        return array[0:n]
    return sample_from_disk(r, n)


def sample_from_domain(n):
    """
    For simplicity, consider a square with a hole.
    Square: [-1,1]*[-1,1]
    Hole: c = (0.3,0.0), r = 0.3
    Args:
        n: Number of samples.
    """
    array = np.zeros([n, 2])
    c = np.array([0.3, 0.0])
    r = 0.3
    for i in range(n):
        array[i] = random_point(c, r)
    return array


def random_point(c, r):
    point = np.random.rand(2) * 2 - 1
    if np.linalg.norm(point - c) < r:
        return random_point(c, r)
    return point


def sample_from_boundary(n):
    """
    For simplicity, consider a square with a hole.
    Square: [-1,1]*[-1,1]
    Hole: c = (0.3,0.0), r = 0.3
    Args:
        n: Number of samples.
    """
    c = np.array([0.3, 0.0])
    r = 0.3
    length = 4 * 2 + 2 * math.pi * r
    interval1 = np.array([0.0, 2.0 / length])
    interval2 = np.array([2.0 / length, 4.0 / length])
    interval3 = np.array([4.0 / length, 6.0 / length])
    interval4 = np.array([6.0 / length, 8.0 / length])
    interval5 = np.array([8.0 / length, 1.0])
    array = np.zeros([n, 2])
    for i in range(n):
        rand0 = np.random.rand()
        rand1 = np.random.rand()
        point1 = np.array([rand1 * 2.0 - 1.0, -1.0])
        point2 = np.array([rand1 * 2.0 - 1.0, +1.0])
        point3 = np.array([-1.0, rand1 * 2.0 - 1.0])
        point4 = np.array([+1.0, rand1 * 2.0 - 1.0])
        point5 = np.array([c[0] + r * math.cos(2 * math.pi * rand1), c[1] + r * math.sin(2 * math.pi * rand1)])
        array[i] = my_fun(rand0, interval1) * point1 + my_fun(rand0, interval2) * point2 + \
                   my_fun(rand0, interval3) * point3 + my_fun(rand0, interval4) * point4 + \
                   my_fun(rand0, interval5) * point5
    return array


def my_fun(x, interval):
    if interval[0] <= x <= interval[1]:
        return 1.0
    return 0.0


def sample_from_surface(r, n):
    """
    Sample points from surface.
    Args:
        r: Radius;
        n: Number of samples.
    """
    array = np.random.normal(size=(n, 2))
    norm = np.linalg.norm(array, 2, axis=1)
    if np.min(norm) == 0:
        return sample_from_surface(r, n)
    array = np.multiply(array.T, 1 / norm).T
    return array * r


def sample_from_disk_10(r, n):
    """
    Sample from 10d-ball.
    Args:
        r: Radius;
        n: Number of samples.
    """
    array = np.random.normal(size=(n, 10))
    norm = np.linalg.norm(array, 2, axis=1)
    if np.min(norm) == 0:
        return sample_from_disk_10(r, n)
    array = np.multiply(array.T, 1 / norm).T
    radius = np.random.rand(n, 1) ** (1 / 10)
    array = np.multiply(array, radius)
    return r * array


def sample_from_surface_10(r, n):
    """
    Sample points from surface.
    Args:
        r: Radius.
        n: Number of samples.
    """
    array = np.random.normal(size=(n, 10))
    norm = np.linalg.norm(array, 2, axis=1)
    if np.min(norm) == 0:
        return sample_from_surface_10(r, n)
    array = np.multiply(array.T, 1 / norm).T
    return array * r
