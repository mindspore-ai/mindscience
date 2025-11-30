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
# ============================================================================
"""
wigner file
"""

# pylint: disable=C0103
import os
import pickle
from mindspore import ops
import mindspore as ms
from mindscience.e3nn.utils.func import broadcast_args

jd = None
file_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(file_dir, 'jd.pkl')

with open(pkl_path, 'rb') as f:
    jd = pickle.load(f)

def wigner_D(lv, alpha, beta, gamma):
    """
    wigner_D function that complies with mindspore.jit compilation
    """
    if not lv < len(jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(jd) - 1}, send us an email to ask for more"
        )
    alpha, beta, gamma = broadcast_args(alpha, beta, gamma)
    j = jd[lv]
    xa = _z_rot_mat(alpha, lv)
    xb = _z_rot_mat(beta, lv)
    xc = _z_rot_mat(gamma, lv)
    return xa @ j.astype(ms.float16) @ xb @ j.astype(ms.float16) @ xc


def _z_rot_mat(angle, lv):
    shape = angle.shape
    m = ops.zeros((shape[0], 2 * lv + 1, 2 * lv + 1))
    inds = ops.arange(0, 2 * lv + 1, 1)
    reversed_inds = ops.arange(2 * lv, -1, -1)
    frequencies = ops.arange(lv, -lv - 1, -1)
    m[..., inds, reversed_inds] = ops.sin(frequencies * angle[..., None])
    m[..., inds, inds] = ops.cos(frequencies * angle[..., None])
    return m.astype(ms.float16)
