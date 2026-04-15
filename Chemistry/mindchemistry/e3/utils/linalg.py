# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import ops


def _direct_sum(*matrices):
    r"""Direct sum of matrices, put them in the diagonal
    """
    front_indices = matrices[0].shape[:-2]
    m = sum(x.shape[-2] for x in matrices)
    n = sum(x.shape[-1] for x in matrices)
    total_shape = list(front_indices) + [m, n]
    zeros = ops.Zeros()
    out = zeros(tuple(total_shape), matrices[0].dtype)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i: i + m, j: j + n] = x
        i += m
        j += n
    return out
