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
# Calculate Positive Pointwise Mutual Information Matrix
"""getPPMImatrix"""
import numpy as np
from scalesimmat import scalesimmat


def getppmimatrix(m):
    """GetPPMIMatrix"""
    m = scalesimmat(m)
    p = m.shape[0]
    q = m.shape[1]
    assert p == q
    col = np.sum(m, 0)
    row = np.sum(m, 1)
    d = np.sum(col)
    row = np.transpose([row])
    m1 = np.matmul(row, [col])
    ppmi = np.log(d * m / m1)
    ppmi = np.where(ppmi > 0, ppmi, 0)
    ppmi = np.where(np.isnan(ppmi), 0, ppmi)
    return ppmi
