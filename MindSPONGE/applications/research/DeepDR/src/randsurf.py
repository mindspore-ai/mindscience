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
"""Randomly Surf"""
import numpy as np
from scalesimmat import scalesimmat


def randsurf(a, max_step, alpha):
    """RandSurf"""
    a = scalesimmat(a)
    num_nodes = len(a)
    p0 = np.eye(num_nodes, num_nodes, dtype=np.float)
    p = p0
    m = np.zeros((num_nodes, num_nodes), dtype=np.float)

    for _ in range(0, max_step):
        p = alpha * np.matmul(p, a) + (1 - alpha) * p0
        m = m + p
    return m
