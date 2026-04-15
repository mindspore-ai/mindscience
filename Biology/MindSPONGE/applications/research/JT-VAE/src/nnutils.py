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
"""nnutils"""
import mindspore.nn as nn
import mindspore.ops as ops


def index_select_nd(source, dim, index):
    index_size = index.shape
    suffix_dim = source.shape[1:]
    final_size = index_size + suffix_dim
    target = source.take(index.view(-1), axis=dim)
    return target.view(final_size)


def gru(x, h_nei, w_z, w_r, u_r, w_h):
    """gru"""
    hidden_size = x.shape[-1]
    sum_h = h_nei.sum(axis=1)
    z_input = ops.concat([x, sum_h], 1)
    z = nn.Sigmoid()(w_z(z_input))

    r_1 = w_r(x).view(-1, 1, hidden_size)
    r_2 = u_r(h_nei)
    r = nn.Sigmoid()(r_1 + r_2)

    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(axis=1)
    h_input = ops.concat([x, sum_gated_h], 1)
    pre_h = nn.Tanh()(w_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h
