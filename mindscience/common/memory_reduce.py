# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""utils module"""

from mindspore.ops import operations as P
from mindspore.ops import functional as F
def _memory_reduce(body, batched_inputs, nonbatched_inputs, slice_num, dim=0):
    """memory reduce function"""
    if slice_num <= 1:
        inputs = batched_inputs + nonbatched_inputs
        return body(*inputs)
    inner_batched_inputs = []
    for val in batched_inputs:
        inner_val = P.Split(dim, slice_num)(val)
        inner_batched_inputs.append(inner_val)
    # for depend
    inner_split_batched_inputs = ()
    for _, inner_batched_input in enumerate(inner_batched_inputs):
        inner_split_batched_inputs = inner_split_batched_inputs + (inner_batched_input[0],)
    inner_split_inputs = inner_split_batched_inputs + nonbatched_inputs
    inner_split_res = body(*inner_split_inputs)
    res = (inner_split_res,)
    for i in range(1, slice_num):
        inner_split_batched_inputs = ()
        for _, inner_batched_input in enumerate(inner_batched_inputs):
            inner_split_batched_inputs = inner_split_batched_inputs + (inner_batched_input[i],)
        inner_split_inputs = inner_split_batched_inputs + nonbatched_inputs
        inner_split_inputs = F.depend(inner_split_inputs, res[-1])
        inner_split_res = body(*inner_split_inputs)
        res = res + (inner_split_res,)
    res = P.Concat()(res)
    return res
