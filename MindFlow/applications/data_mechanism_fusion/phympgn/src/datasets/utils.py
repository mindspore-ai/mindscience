# Copyright 2025 Huawei Technologies Co., Ltd
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
"""datasets utils"""
import enum

import mindspore.numpy as mnp
from mindspore import ops


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    INLET = 2
    OUTLET = 3


def add_noise(truth, percentage=0.05):
    """add noise"""
    # shape of truth must be (n, 2)
    assert truth.shape[1] == 2
    uv = [truth[:, 0:1], truth[:, 1:2]]
    uv_noi = []
    for component in uv:
        r = ops.normal(mean=0.0, stddev=1.0, shape=component.shape)
        std_r = ops.std(r)          # std of samples
        std_t = ops.std(component)
        noise = r * std_t / std_r * percentage
        uv_noi.append(component + noise)
    return ops.cat(uv_noi, axis=1)


def to_undirected(edge_index, num_nodes):
    """to undirected"""
    row, col = edge_index[0], edge_index[1]
    row, col = ops.cat([row, col], axis=0), ops.cat([col, row], axis=0)
    edge_index = ops.stack([row, col], axis=0)

    return coalesce(edge_index, num_nodes)


def coalesce(edge_index, num_nodes, is_sorted=False, sort_by_row=True):
    """coalesce"""
    nnz = edge_index.shape[1]
    idx = mnp.empty(nnz + 1, dtype=edge_index.dtype)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:] = idx[1:].mul(num_nodes).add(edge_index[int(sort_by_row)])

    if not is_sorted:
        # idx[1:], perm = index_sort(idx[1:], max_value=num_nodes * num_nodes)
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        return edge_index

    edge_index = edge_index[:, mask]

    return edge_index
