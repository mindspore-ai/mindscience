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
"""phympgn"""
from mindspore import nn, ops
import mindspore as ms

from .encoder_decoder import Encoder, Decoder
from .mpnn_block import MPNNBlock
from .laplace_block import LaplaceBlock
from ..utils.padding import graph_padding

class PhyMPGN(nn.Cell):
    """PhyMPGN"""
    def __init__(self, encoder_config, mpnn_block_config, decoder_config,
                 laplace_block_config, integral):
        super().__init__()
        self.node_encoder = Encoder(encoder_config['node_encoder_layers'])
        self.edge_encoder = Encoder(encoder_config['edge_encoder_layers'])
        self.mpnn_block = MPNNBlock(
            mpnn_layers=mpnn_block_config['mpnn_layers'],
            mpnn_num=mpnn_block_config['mpnn_num']
        )
        self.decoder = Decoder(decoder_config['node_decoder_layers'])
        self.laplace_block = LaplaceBlock(
            enc_dim=laplace_block_config['in_dim'],
            h_dim=laplace_block_config['h_dim'],
            out_dim=laplace_block_config['out_dim']
        )

        update_fn = {
            1: self.update_euler,
            2: self.update_rk2,
            4: self.update_rk4
        }
        self.update = update_fn[integral]

    def construct(self, graph, steps):
        """
        Args:
            graph (Graph): instance of Graph, involving edge_index, pos, y, and etc.
            steps (int): steps of roll-out

        Returns:
            loss_states (Tensor): predicted states
        """
        loss_states = [graph.y] # [bn, 2]
        # unroll for 1 step
        graph_next = self.update(graph)
        loss_states.append(graph_next.y)

        graph = graph_next.detach()
        # unroll for steps-1
        for _ in range(steps - 1):
            graph_next = self.update(graph)
            loss_states.append(graph_next.y)
            graph = graph_next

        # [t, bn, 2]
        loss_states = ops.stack(loss_states, axis=0)
        return ops.index_select(loss_states, 1, graph.truth_index)

    def get_temporal_diff(self, graph):
        """compute results of F nonlinear operator"""
        node_type = ops.one_hot(graph.node_type, graph.node_type.max() + 1)
        node_type = node_type.astype(ms.float32)
        graph.state_node = self.node_encoder(
            ops.cat((graph.y, graph.pos, node_type), axis=-1))
        # store dirichlet value
        if hasattr(graph, 'dirichlet_index'):
            graph.dirichlet_h_value = ops.index_select(
                graph.state_node, 0, graph.dirichlet_index)
            graph.inlet_h_value = ops.index_select(
                graph.state_node, 0, graph.inlet_index)

        rel_state = graph.y[graph.edge_index[1, :]] - \
                    graph.y[graph.edge_index[0, :]]  # (b_e, 2)
        # (b_e, 5) -> (b_e, h)
        graph.state_edge = self.edge_encoder(
            ops.cat((rel_state, graph.edge_attr), axis=-1))
        mpnn_out = self.mpnn_block(graph)  # (b_e, h)
        decoder_out = self.decoder(mpnn_out)  # (b_n, 2)

        # laplace
        laplace = self.laplace_block(graph)  # (b_n, 2)

        u_m, rho, d, mu = graph.u_m, graph.rho, graph.r * 2, graph.mu
        re = rho * d * u_m / mu  # (b_n, 1)
        # (b_n, 1) * (b_n, 2) + (b_n, 2) -> (b_n, 2)
        out = 1 / re * laplace + decoder_out

        return out

    def update_euler(self, graph):
        """euler scheme"""
        out = self.get_temporal_diff(graph)
        graph.y = graph.y + out * graph.dt
        # padding
        graph_padding(graph)

        return graph

    def update_rk2(self, graph):
        """rk2 scheme"""
        u0 = graph.y
        k1 = self.get_temporal_diff(graph)  # (bn, 2)
        u1 = u0 + k1 * graph.dt  # (bn, 2) + (bn, 2) * (bn, 1) -> (bn, 2)
        graph.y = u1
        # padding
        graph_padding(graph)

        k2 = self.get_temporal_diff(graph)
        graph.y = u0 + k1 * graph.dt / 2 + k2 * graph.dt / 2
        # padding
        graph_padding(graph)

        return graph

    def update_rk4(self, graph):
        """rk4 scheme"""
        # stage 1
        u0 = graph.y
        k1 = self.get_temporal_diff(graph)

        # stage 2
        u1 = u0 + k1 * graph.dt / 2.
        graph.y = u1
        # padding
        graph_padding(graph)
        k2 = self.get_temporal_diff(graph)

        # stage 3
        u2 = u0 + k2 * graph.dt / 2.
        graph.y = u2
        # padding
        graph_padding(graph)
        k3 = self.get_temporal_diff(graph)

        # stage 4
        u3 = u0 + k3 * graph.dt
        graph.y = u3
        # padding
        graph_padding(graph)
        k4 = self.get_temporal_diff(graph)

        u4 = u0 + (k1 + 2 * k2 + 2 * k3 + k4) * graph.dt / 6.
        graph.y = u4
        # padding
        graph_padding(graph)

        return graph

    @property
    def num_params(self):
        total = sum(param.size for param in self.trainable_params())
        mpnn = self.mpnn_block.num_params
        laplace = self.laplace_block.num_params
        return total, mpnn, laplace
