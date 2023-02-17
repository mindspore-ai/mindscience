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
"""Some functions used in GVP"""

import mindspore as ms
import mindspore.ops as ops


def ms_flatten(input_tensor, start_dim, end_dim):
    """Flatten"""
    if start_dim == 0:
        shape_list = list(input_tensor.shape[end_dim + 1:])
        dim = 1
        for i in range(start_dim, end_dim + 1):
            dim = input_tensor.shape[i] * dim
        shape_list.insert(0, dim)
        shape_list = tuple(shape_list)
        flatten = ms.ops.Reshape()
        output = flatten(input_tensor, shape_list)
        return output
    if end_dim in (-1, input_tensor.dim() - 1):
        shape_list = list(input_tensor.shape[:start_dim])
        dim = 1
        for i in range(start_dim, end_dim + 1):
            dim = input_tensor.shape[i] * dim
        shape_list.append(dim)
        shape_list = tuple(shape_list)
        flatten = ms.ops.Reshape()
        output = flatten(input_tensor, shape_list)
        return output
    raise ValueError("Unknown dim selected")


def flatten_graph(node_embeddings, edge_embeddings, edge_index):
    """Flatten graph"""
    x_s, x_v = node_embeddings
    e_s, e_v = edge_embeddings
    batch_size, n = x_s.shape[0], x_s.shape[1]
    node_embeddings = (x_s.reshape(((x_s.shape[0] * x_s.shape[1]), x_s.shape[2])),
                       x_v.reshape(((x_v.shape[0] * x_v.shape[1]), x_v.shape[2], x_v.shape[3])))
    edge_embeddings = (e_s.reshape(((e_s.shape[0] * e_s.shape[1]), e_s.shape[2])),
                       e_v.reshape(((e_v.shape[0] * e_v.shape[1]), e_v.shape[2], e_v.shape[3])))
    new_edge_index = ops.Cast()(edge_index != -1, ms.bool_)
    edge_mask = new_edge_index.any(axis=1)

    # Re-number the nodes by adding batch_idx * N to each batch
    unsqueeze = ops.ExpandDims()
    edge_index = edge_index + unsqueeze(unsqueeze((ms.numpy.arange(batch_size) * n), -1), -1)

    permute = ops.Transpose()

    edge_index = permute(edge_index, (1, 0, 2))
    edge_index = edge_index.reshape(edge_index.shape[0], (edge_index.shape[1] * edge_index.shape[2]))

    edge_mask = edge_mask.flatten()
    edge_mask = edge_mask.asnumpy()
    edge_index = edge_index.asnumpy()
    edge_embeddings_0 = edge_embeddings[0].asnumpy()
    edge_embeddings_1 = edge_embeddings[1].asnumpy()

    edge_index = edge_index[:, edge_mask]
    edge_embeddings = (
        ms.Tensor(edge_embeddings_0[edge_mask, :], ms.float32),
        ms.Tensor(edge_embeddings_1[edge_mask, :], ms.float32)
    )

    edge_index = ms.Tensor(edge_index, ms.int32)
    return node_embeddings, edge_embeddings, edge_index


def unflatten_graph(node_embeddings, batch_size):
    """Unflatten graph"""
    x_s, x_v = node_embeddings
    x_s = x_s.reshape((batch_size, -1, x_s.shape[1]))
    x_v = x_v.reshape((batch_size, -1, x_v.shape[1], x_v.shape[2]))
    return (x_s, x_v)
