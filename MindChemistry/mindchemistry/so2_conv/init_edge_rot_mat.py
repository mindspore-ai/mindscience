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
file to get rotating matrix from edge distance vector
"""
from mindspore import ops
import mindspore.numpy as ms_np


def init_edge_rot_mat(edge_distance_vec):
    """
    get rotating matrix from edge distance vector
    """
    epsilon = 0.00000001
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = ops.sqrt(ops.maximum(ops.sum(edge_vec_0 ** 2, dim=1), epsilon))
    # Make sure the atoms are far enough apart
    norm_x = ops.div(edge_vec_0, edge_vec_0_distance.view(-1, 1))
    edge_vec_2 = ops.rand_like(edge_vec_0) - 0.5

    edge_vec_2 = ops.div(edge_vec_2, ops.sqrt(ops.maximum(ops.sum(edge_vec_2 ** 2, dim=1), epsilon)).view(-1, 1))
    # Create two rotated copies of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.copy()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.copy()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = ops.abs(ops.sum(edge_vec_2b * norm_x, dim=1)).view(-1, 1)
    vec_dot_c = ops.abs(ops.sum(edge_vec_2c * norm_x, dim=1)).view(-1, 1)
    vec_dot = ops.abs(ops.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = ops.where(ops.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2)
    vec_dot = ops.abs(ops.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = ops.where(ops.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2)
    vec_dot = ops.abs(ops.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned

    norm_z = ms_np.cross(norm_x, edge_vec_2, axis=1)
    norm_z = ops.div(norm_z, ops.sqrt(ops.maximum(ops.sum(norm_z ** 2, dim=1, keepdim=True), epsilon)))
    norm_z = ops.div(norm_z, ops.sqrt(ops.maximum(ops.sum(norm_z ** 2, dim=1), epsilon)).view(-1, 1))

    norm_y = ms_np.cross(norm_x, norm_z, axis=1)
    norm_y = ops.div(norm_y, ops.sqrt(ops.maximum(ops.sum(norm_y ** 2, dim=1, keepdim=True), epsilon)))
    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)
    edge_rot_mat_inv = ops.cat([norm_z, norm_x, norm_y], axis=2)

    edge_rot_mat = ops.swapaxes(edge_rot_mat_inv, 1, 2)
    return edge_rot_mat
