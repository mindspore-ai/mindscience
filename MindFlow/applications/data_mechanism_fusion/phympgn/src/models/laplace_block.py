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
"""Laplace block"""
from mindspore import nn, ops

from .utils import activation_func
from .mpnn_layer import MPNNLayer


class LaplaceBlock(nn.Cell):
    """Laplace block"""
    def __init__(self, enc_dim, h_dim, out_dim):
        super().__init__()
        self.encoder = nn.SequentialCell(
            nn.Dense(enc_dim, h_dim),
            activation_func(),
            nn.Dense(h_dim, h_dim)
        )
        self.processor = LaplaceProcessor(
            mpnn_layers=[
                [h_dim * 2 + 3, h_dim, h_dim],
                [h_dim * 2, h_dim, h_dim]
            ],
            mpnn_num=3
        )
        self.decoder = nn.SequentialCell(
            nn.Dense(h_dim, h_dim),
            activation_func(),
            nn.Dense(h_dim, out_dim)
        )

    def cal_mesh_laplace(self, graph):
        laplace = graph.L @ graph.y
        return laplace

    def construct(self, graph):
        h = self.encoder(ops.cat((graph.y, graph.pos), axis=-1))
        edge_attr = graph.edge_attr[:, :3]
        h = self.processor(h, edge_attr, graph.edge_index)
        out = self.decoder(h)
        out = graph.d * out

        out = out + self.cal_mesh_laplace(graph)
        return out

    @property
    def num_params(self):
        params = self.trainable_params()
        return sum(param.size for param in params)


class LaplaceProcessor(nn.Cell):
    """Laplace Processor"""
    def __init__(self, mpnn_layers, mpnn_num):
        super().__init__()
        self.phi_layers = mpnn_layers[0]
        self.gamma_layers = mpnn_layers[1]
        self.mpnn_num = mpnn_num
        self.nets = self.build_block()

    def build_block(self):
        nets = nn.CellList()
        for _ in range(self.mpnn_num):
            nets.append(MPNNLayer(self.phi_layers, self.gamma_layers))
        return nets

    def construct(self, h, edge_attr, edge_index):
        for mpnn in self.nets:
            h = h + mpnn(edge_index=edge_index,
                         node_features=h,
                         edge_features=edge_attr)
        return h
