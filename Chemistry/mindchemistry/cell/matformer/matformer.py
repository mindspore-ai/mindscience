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
"""Matformer"""
from mindspore import nn, ops
import mindspore as ms
from mindchemistry.graph.graph import AggregateNodeToGlobal
from mindchemistry.cell.matformer.utils import RBFExpansion
from mindchemistry.cell.matformer.transformer import MatformerConv, Silu


class Matformer(nn.Cell):
    """Matformer class"""

    def __init__(self, config):
        """init"""
        super(Matformer, self).__init__()

        use_fp16 = config['use_fp16']
        self.classification = config['classification']
        self.use_angle = config['use_angle']

        self.global_dtype = ms.float32
        if use_fp16:
            self.global_dtype = ms.float16

        self.atom_embedding = nn.Dense(
            config['atom_input_features'], config['node_features']
        ).to_float(self.global_dtype)

        ##################
        self.rbf_0 = RBFExpansion(vmin=0, vmax=8.0, bins=config['edge_features'],)
        self.rbf_1 = nn.Dense(config['edge_features'], config['node_features']).to_float(self.global_dtype)
        self.rbf_2 = ops.Softplus()
        self.rbf_3 = nn.Dense(config['node_features'], config['node_features']).to_float(self.global_dtype)

        self.angle_lattice = config["angle_lattice"]

        self.att_layers = nn.CellList(
            [
                MatformerConv(in_channels=config['node_features'], out_channels=config['node_features'],
                              heads=config['node_layer_head'], edge_dim=config['node_features'])
                for _ in range(config['conv_layers'])
            ]
        )

        self.fc = nn.SequentialCell(
            nn.Dense(config['node_features'], config['fc_features']).to_float(self.global_dtype),
            Silu().to_float(ms.float32)
        )

        self.fc_out = nn.Dense(
            config['fc_features'], config['output_features']
        )

        self.link = None
        self.link_name = config['link']
        if config['link'] == "identity":
            self.link = lambda x: x

        self.dim_size = config["batch_size_max"]
        self.aggregator_to_global = AggregateNodeToGlobal("mean")

    def construct(self, data_x, data_edge_attr, data_edge_index, data_batch, node_mask, edge_mask, node_num):
        """construct"""
        node_features = self.atom_embedding(data_x)
        edge_feat = ops.norm(data_edge_attr, dim=1)
        edge_features = self.rbf_0(edge_feat)
        edge_features = ops.mul(edge_features, ops.reshape(edge_mask, (-1, 1)))
        edge_features = self.rbf_1(edge_features)
        edge_features = self.rbf_2(edge_features)
        edge_features = self.rbf_3(edge_features)

        node_features = self.att_layers[0](node_features, data_edge_index, edge_features, node_mask, edge_mask,
                                           node_num)
        node_features = self.att_layers[1](node_features, data_edge_index, edge_features, node_mask, edge_mask,
                                           node_num)
        node_features = self.att_layers[2](node_features, data_edge_index, edge_features, node_mask, edge_mask,
                                           node_num)
        node_features = self.att_layers[3](node_features, data_edge_index, edge_features, node_mask, edge_mask,
                                           node_num)
        node_features = self.att_layers[4](node_features, data_edge_index, edge_features, node_mask, edge_mask,
                                           node_num)

        features = self.aggregator_to_global(node_features, data_batch, dim_size=self.dim_size, mask=node_mask)

        features = self.fc(features)
        out = self.fc_out(features)
        res = ops.squeeze(out)
        return res
