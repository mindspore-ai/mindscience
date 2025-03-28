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
"""data loader"""
import numpy as np
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor
import mindspore as ms

from ..datasets.data import Graph
from ..datasets.dataset import PDECFDataset


def get_data_loader(dataset: PDECFDataset, batch_size: int, shuffle: bool = True) \
        -> ds.GeneratorDataset:
    """get data loader"""
    column_names = ['pos', 'y', 'edge_index', 'edge_attr', 'dt', 'mu', 'r',
                    'rho', 'L', 'd', 'u_m', 'dirichlet_index', 'inlet_index',
                    'dirichlet_value', 'inlet_value', 'node_type',
                    'truth_index']

    if shuffle:
        loader = ds.GeneratorDataset(
            source=dataset,
            column_names=column_names
        )
        loader = loader.shuffle(buffer_size=5000).batch(batch_size)
    else:
        loader = ds.GeneratorDataset(
            source=dataset,
            column_names=column_names,
            shuffle=False
        )
        loader = loader.batch(batch_size)
    return loader


def batch_graph(data: dict):
    """batch graph"""
    pos, y, edge_index = data['pos'], data['y'], data['edge_index']
    edge_attr = data['edge_attr']
    dt, mu, r, rho, node_type, u_m = data['dt'], data['mu'], data['r'], data['rho'], \
        data['node_type'], data['u_m']
    l, d = data['L'], data['d']
    dirichlet_index, inlet_index = data['dirichlet_index'], data['inlet_index']
    dirichlet_value, inlet_value = data['dirichlet_value'], data['inlet_value']
    truth_index = data['truth_index']

    batch_size = pos.shape[0]
    node_num = pos.shape[1]
    edge_num = edge_index.shape[2]
    m = y.shape[2]
    # (b, n, p_d) -> (b*n, p_d)
    pos_batch = pos.reshape(batch_size * node_num, pos.shape[2])
    # (b, n, m, y_d) -> (b*n, m, y_d)
    y_batch = y.reshape(batch_size * node_num, m, y.shape[3])
    # (b, 2, e) -> (2, b_e)
    edge_index_batch = batched_edge_index(edge_index, node_num)
    # (b, e, e_d) -> (b*e, e_d)
    edge_attr_batch = edge_attr.reshape(batch_size * edge_num,
                                        edge_attr.shape[2])
    # (bn,)
    batch = np.concatenate([i * np.ones(node_num) for i in range(batch_size)])
    batch = Tensor(batch, dtype=ms.int64)

    # (b, n, 1) -> (b*n, 1)
    dt_batch = dt.reshape(batch_size * node_num, dt.shape[2])
    mu_batch = mu.reshape(batch_size * node_num, mu.shape[2])
    r_batch = r.reshape(batch_size * node_num, r.shape[2])
    rho_batch = rho.reshape(batch_size * node_num, rho.shape[2])
    u_m_batch = u_m.reshape(batch_size * node_num, u_m.shape[2])

    # (b, n, n) -> (b*n, b*n)
    l_batch = ops.block_diag(*ops.unbind(l))
    # (b, n, 1) -> (b*n, 1)
    d_batch = d.reshape(batch_size * node_num, d.shape[2])

    # (b, m) -> (b*m,)
    node_type_batch = node_type.reshape(-1)
    dirichlet_index_batch = batched_node_index(dirichlet_index, node_num)
    inlet_index_batch = batched_node_index(inlet_index, node_num)
    truth_index_batch = batched_node_index(truth_index, node_num)

    # (b, m, 2) -> (b*m, 2)
    dirichlet_value_batch = dirichlet_value.reshape(
        batch_size * dirichlet_value.shape[1], dirichlet_value.shape[2])
    inlet_value_batch = inlet_value.reshape(
        batch_size * inlet_value.shape[1], inlet_value.shape[2])

    return Graph(pos=pos_batch, y=y_batch, edge_index=edge_index_batch,
                 edge_attr=edge_attr_batch,
                 dt=dt_batch, mu=mu_batch, r=r_batch, rho=rho_batch,
                 u_m=u_m_batch,
                 node_type=node_type_batch, L=l_batch, d=d_batch,
                 dirichlet_index=dirichlet_index_batch,
                 inlet_index=inlet_index_batch,
                 dirichlet_value=dirichlet_value_batch,
                 inlet_value=inlet_value_batch,
                 truth_index=truth_index_batch,
                 batch=batch)


def batched_edge_index(edge_index, node_num):
    """batch edge index

    Args:
        edge_index (Tensor): Shape (b, 2, e)
        node_num (int): Number of nodes in each graph

    Returns:
        edge_index_batch: batched edge index
    """
    add_index = np.concatenate(
        [node_num*i*np.ones([1, edge_index.shape[1],
                             edge_index.shape[2]], np.int64)
         for i in range(edge_index.shape[0])], axis=0)
    if isinstance(edge_index, np.ndarray):
        return add_index + edge_index
    edge_index_batch = Tensor(add_index) + edge_index
    batch_size, edge_num = edge_index_batch.shape[0], edge_index_batch.shape[2]
    edge_index_batch = ops.permute(edge_index_batch, (1, 0, 2))\
        .reshape(2, edge_num * batch_size)
    return edge_index_batch


def batched_node_index(node_index, node_num):
    """batched index of nodes

    Args:
        node_index (Tensor): Shape (b, m)
        node_num (int): Number of nodes in each graph

    Returns:
        node_index_batch: batched index of nodes.
    """
    add_index = np.concatenate(
        [node_num*i*np.ones([1, node_index.shape[1]], np.int64)
         for i in range(node_index.shape[0])], axis=0)
    if isinstance(node_index, np.ndarray):
        return add_index + node_index
    node_index_batch = Tensor(add_index) + node_index
    node_index_batch = node_index_batch.reshape(-1)
    return node_index_batch
