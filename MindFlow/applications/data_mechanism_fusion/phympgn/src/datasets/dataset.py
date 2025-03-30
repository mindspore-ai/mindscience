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
"""dataset"""
import os.path as osp

import h5py
from tqdm import tqdm
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops

from .transform import Compose, Distance, Cartesian, Dirichlet, \
    DirichletInlet, NodeTypeInfo, MaskFace, Delaunay, FaceToEdge
from .data import Graph
from .utils import add_noise
from ..utils.voronoi_laplace import compute_discrete_laplace
from ..utils.padding import graph_padding


class PDECFDataset:
    """PDECFDataset"""
    def __init__(self, root, raw_files, dataset_start,
                 dataset_used, time_start, time_used, window_size,
                 training=False):
        self.raw_files = raw_files
        self.laplace_file = 'laplace.npy'
        self.d_file = 'd_vector.npy'
        self.root = root
        self.training = training

        self.dataset_start = dataset_start
        self.dataset_used = dataset_used
        self.time_start = time_start
        self.time_used = time_used
        self.window_size = window_size

        self.set_transform()
        self.data_list = self.process()

    def __getitem__(self, index):
        graph = self.data_list[index]
        return graph.pos, graph.y, graph.edge_index, graph.edge_attr, graph.dt, \
            graph.mu, graph.r, graph.rho, graph.L, graph.d, graph.u_m, \
            graph.dirichlet_index, graph.inlet_index, graph.dirichlet_value, \
            graph.inlet_value, graph.node_type, graph.truth_index

    def __len__(self):
        return len(self.data_list)

    def set_transform(self):
        """set transform"""
        self.periodic_trans = None
        self.dirichlet_trans = Dirichlet()
        self.inlet_trans = DirichletInlet()
        self.neumann_trans = None
        self.node_type_trans = NodeTypeInfo()
        self.mask_face_trans = MaskFace()
        self.transform = [
            Delaunay(),
            self.mask_face_trans,
            FaceToEdge(remove_faces=False),
            Distance(norm=True),
            Cartesian(norm=True),
        ]
        if self.dirichlet_trans is not None:
            self.transform.append(self.dirichlet_trans)
        if self.dirichlet_trans is not None:
            self.transform.append(self.inlet_trans)
        if self.periodic_trans is not None:
            self.transform.append(self.periodic_trans)
        if self.neumann_trans is not None:
            self.transform.append(self.neumann_trans)
        self.transform.append(self.node_type_trans)
        self.transform = Compose(transforms=self.transform)

    def process(self):
        """process data"""
        data_list = []
        file_handler = h5py.File(osp.join(self.root, self.raw_files))
        coarse_pos = file_handler['pos'][:]  # (n, 2)
        r = file_handler.attrs['r']
        mu = file_handler.attrs['mu']
        rho = file_handler.attrs['rho']
        node_type = file_handler['node_type']
        inlet_index, cylinder_index = node_type['inlet'][:], node_type['cylinder'][:]
        self.dirichlet_trans.set_index(cylinder_index)
        self.inlet_trans.set_index(inlet_index)
        self.node_type_trans.set_type_dict(node_type)
        self.mask_face_trans.set_cylinder_index(cylinder_index)
        for i in tqdm(range(self.dataset_start, self.dataset_used)):
            # (t, n_f, d)
            g = file_handler[str(i)]
            u = g['U'][:]
            dt = g.attrs['dt']
            u_m = g.attrs['u_m']

            # dimensionless
            u = u / u_m
            pos = coarse_pos / (2 * r)
            dt = dt / (2 * r / u_m)

            # to tensor
            u_t = Tensor(u, dtype=ms.float32)  # (t, n, d)
            pos_t = Tensor(pos, dtype=ms.float32)
            # (n,)
            truth_index = Tensor(ms.numpy.arange(pos.shape[0]), dtype=ms.int64)
            # (n, 1)
            u_m_t = ops.ones((pos.shape[0], 1), dtype=ms.float32) * u_m
            dt_t = ops.ones((pos.shape[0], 1), dtype=ms.float32) * dt
            r_t = ops.ones((pos.shape[0], 1), dtype=ms.float32) * r
            mu_t = ops.ones((pos.shape[0], 1), dtype=ms.float32) * mu
            rho_t = ops.ones((pos.shape[0], 1), dtype=ms.float32) * rho

            for idx in ms.numpy.arange(self.time_start,
                                       self.time_start + self.time_used,
                                       step=self.window_size):
                # [t, n, c] -> [n, t, c]
                if idx + self.window_size > self.time_start + self.time_used:
                    break
                y = u_t[idx:idx + self.window_size].permute(1, 0, 2)
                if self.training:
                    y[:, 0, :] = add_noise(y[:, 0, :], percentage=0.03)

                graph = Graph(pos=pos_t, y=y,
                              truth_index=truth_index,
                              dt=dt_t, u_m=u_m_t,
                              r=r_t, mu=mu_t,
                              rho=rho_t)
                graph = self.transform(graph)
                data_list.append(graph)

        if osp.exists(osp.join(self.root, self.laplace_file)):
            laplace_matrix_np = np.load(osp.join(self.root, self.laplace_file))
            d_vector_np = np.load(osp.join(self.root, self.d_file))
            laplace_matrix = ms.Tensor(laplace_matrix_np, dtype=ms.float32)
            d_vector = ms.Tensor(d_vector_np, dtype=ms.float32)
        else:
            laplace_matrix_np, d_vector_np = compute_discrete_laplace(
                pos=data_list[0].pos.numpy(),
                edge_index=data_list[0].edge_index.numpy(),
                face=data_list[0].face.numpy()
            )
            d_vector_np = d_vector_np[:, None]
            np.save(osp.join(self.root, self.laplace_file), laplace_matrix_np)
            np.save(osp.join(self.root, self.d_file), d_vector_np)
            laplace_matrix = ms.Tensor(laplace_matrix_np, dtype=ms.float32)
            d_vector = ms.Tensor(d_vector_np, dtype=ms.float32)

        for data in data_list:
            data.L = laplace_matrix
            data.d = d_vector
            data.dirichlet_value = ops.zeros((data.dirichlet_index.shape[0],
                                              data.y.shape[2]))
            data.inlet_value = self.inlet_velocity(
                data.inlet_index, 1.)
            graph_padding(data, clone=True)

        return data_list

    @staticmethod
    def inlet_velocity(inlet_index, u_m):
        u = u_m * ops.ones(inlet_index.shape[0])
        v = ops.zeros_like(u)

        return ops.stack((u, v), axis=-1)  # (m, 2)

    @staticmethod
    def dimensional(u_pred, u_gt, pos, u_m, d):
        u_pred = u_pred * u_m
        u_gt = u_gt * u_m
        pos = pos * d

        return u_pred, u_gt, pos
