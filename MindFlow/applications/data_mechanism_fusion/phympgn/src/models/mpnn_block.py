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
"""mpnn block"""
from mindspore import nn

from ..utils.padding import h_padding
from .mpnn_layer import MPNNLayer


class MPNNBlock(nn.Cell):
    """MPNNBlock"""
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

    def construct(self, graph):
        """construct"""
        h = graph.state_node
        for i in range(len(self.nets) - 1):
            mpnn = self.nets[i]
            h = h + mpnn(
                edge_index=graph.edge_index,
                node_features=h,
                edge_features=graph.state_edge
            )
            # padding
            h_padding(h, graph)

        h = self.nets[-1](
            edge_index=graph.edge_index,
            node_features=h,
            edge_features=graph.state_edge
        )
        return h  # (b_n, node_h_dim)

    @property
    def num_params(self):
        params = self.trainable_params()
        return sum(param.size for param in params)
