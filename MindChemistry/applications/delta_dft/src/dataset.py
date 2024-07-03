# Copyright 2024 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
data processing and kernel functions
"""
import numpy as np
import mindspore as ms


class Normalize:
    """Normalize"""

    def __init__(self, mean_std_params, data_type):
        self.mean_std_params = mean_std_params
        mean_std = {
            'density': tuple(mean_std_params['mean_std_density']),
            'energy': tuple(mean_std_params['mean_std_energy']),
        }
        mean, std = mean_std[data_type]
        self.mean = mean
        self.std = std

    def apply_to_data(self, dist: ms.Tensor) -> ms.Tensor:
        """dist: (N,N)"""
        if self.mean is None:
            self.mean = ms.ops.mean(dist)
            self.std = ms.ops.std(dist)
        return (dist - self.mean) / self.std

    def recover(self, y):
        return y * self.std + self.mean


class LaplacianKernel:
    """Laplacian核"""

    def __init__(self, gamma=0):
        self.gamma = gamma if gamma is not None else 1.0

    def apply(self, x1: ms.Tensor, x2: ms.Tensor, gamma=None) -> ms.Tensor:
        if gamma is None:
            gamma = self.gamma
        return self.apply_to_dist(dist=ms.ops.norm(x1 - x2), gamma=gamma)

    def apply_to_dist(self, dist: ms.Tensor, gamma=None) -> ms.Tensor:
        if gamma is None:
            gamma = self.gamma
        kernel = ms.ops.exp(-ms.ops.abs(dist) / gamma)
        return kernel


class RBFKernel:
    """RBF核"""

    def __init__(self, gamma=0):
        self.gamma = gamma

    def apply(self, x1: ms.Tensor, x2: ms.Tensor, gamma=None) -> ms.Tensor:
        if gamma is None:
            gamma = self.gamma
        return self.apply_to_dist(dist=ms.ops.norm(x1 - x2), gamma=gamma)

    def apply_to_dist(self, dist: ms.Tensor, gamma=None) -> ms.Tensor:
        if gamma is None:
            gamma = self.gamma
        kernel = ms.ops.exp(-ms.ops.pow(dist, 2) / (2 * gamma ** 2))
        return kernel


class MaternKernel:
    """Matérn核"""

    def __init__(self, gamma=1, n=1):
        self.n = n if n is not None else 1
        self.gamma = gamma if gamma is not None else 1.0

    def apply(self, x1: ms.Tensor, x2: ms.Tensor, gamma=None, n=None) -> ms.Tensor:
        if gamma is None:
            gamma = self.gamma
        if n is None:
            n = self.n

        norm_ab = ms.ops.norm(x1 - x2)
        k = self.apply_to_dist(dist=norm_ab, gamma=gamma, n=n)
        return k

    def apply_to_dist(self, dist: ms.Tensor, gamma=None, n=None) -> ms.Tensor:
        """apply to dist of matern kernel"""
        if gamma is None:
            gamma = self.gamma
        if n is None:
            n = self.n

        v = n + 1 / 2
        kernel = ms.ops.mul(ms.ops.exp(ms.ops.mul(dist, -np.sqrt(2 * v)) / gamma),
                            np.math.factorial(n + 1) / np.math.factorial(2 * n + 1))

        s = 0
        for i in range(0, n + 1):
            s = s + ms.ops.mul(ms.ops.pow(ms.ops.mul(dist, np.sqrt(8 * v)) / gamma, n - i),
                               np.math.factorial(n + i) / (np.math.factorial(i) * np.math.factorial(n - i)))
        kernel = ms.ops.mul(kernel, s)

        return kernel


def shuffle(x):
    """shuffle"""
    test_number = int(x.shape[0] * 0.1)
    shuffled_indices = np.random.permutation(x.shape[0])
    train_indices = shuffled_indices[:-test_number].tolist()
    test_indices = shuffled_indices[-test_number:].tolist()
    return train_indices, test_indices
