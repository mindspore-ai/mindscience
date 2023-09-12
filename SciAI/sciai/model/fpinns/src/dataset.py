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
"""random points sampling"""
import numpy as np
import skopt


def generate_sample(n_samples):
    # Certain points should be removed:
    # - Boundary points such as [..., 0, ...]
    # - Special points [0, 0, 0, ...] and [0.5, 0.5, 0.5, ...], which cause error in
    #   Hypersphere.random_points() and Hypersphere.random_boundary_points()
    # 1st point: [0, 0, ...]
    sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
    space = [(0.0, 1.0)]
    return np.asarray(
        sampler.generate(dimensions=space, n_samples=n_samples)[0:], dtype=np.float32
    )


def random_points(n, diam, l):
    x = generate_sample(n)
    return (diam * x + l).astype(np.float32)
