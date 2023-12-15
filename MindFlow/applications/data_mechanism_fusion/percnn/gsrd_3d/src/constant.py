# ============================================================================
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
"""constants"""
import numpy as np

lap_3d_op = np.zeros((1, 1, 5, 5, 5))
elements = [
    (-15/2, (0, 0, 0)),
    (4 / 3, (1, 0, 0)),
    (4 / 3, (0, 1, 0)),
    (4 / 3, (0, 0, 1)),
    (4 / 3, (-1, 0, 0)),
    (4 / 3, (0, -1, 0)),
    (4 / 3, (0, 0, -1)),
    (-1 / 12, (-2, 0, 0)),
    (-1 / 12, (0, -2, 0)),
    (-1 / 12, (0, 0, -2)),
    (-1 / 12, (2, 0, 0)),
    (-1 / 12, (0, 2, 0)),
    (-1 / 12, (0, 0, 2)),
]
for weight, (x, y, z) in elements:
    lap_3d_op[0, 0, x+2, y+2, z+2] = weight
