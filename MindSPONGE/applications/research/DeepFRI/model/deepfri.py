# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""model"""
import mindspore.nn as nn
import mindspore.ops as ops
from module.layers import LstmLayer, MultiGraphConv, SumPooling, FuncPredictor


class DeepFRI(nn.Cell):
    """
        Class for DeepFRI model.
    """

    def __init__(self, input_dim, output_dim, gc_dims: list, fc_dim=1024, drop=0.3, train=False, lstm_input_dim=512):
        super(DeepFRI, self).__init__()
        self.lstm = LstmLayer(lstm_input_dim)
        self.lm_dense = nn.Dense(1024, 1024, has_bias=True)
        self.aa_dense = nn.Dense(input_dim, 1024, has_bias=False)
        self.add = ops.Add()
        self.activation = nn.ReLU()
        self.multi_graph_conv_1 = MultiGraphConv(1024, gc_dims[0], activation='relu', train=train)
        self.multi_graph_conv_2 = MultiGraphConv(gc_dims[0], gc_dims[1], activation='relu', train=train)
        self.multi_graph_conv_3 = MultiGraphConv(gc_dims[1], gc_dims[2], activation='relu', train=train)
        self.cat = ops.Concat(-1)
        self.sum_pooling = SumPooling(1)
        self.en_dense = nn.Dense((gc_dims[0] + gc_dims[1] + gc_dims[2]), fc_dim, has_bias=True, activation='relu')
        self.dropout = nn.Dropout(1 - drop)
        self.func_predictor = FuncPredictor(fc_dim, output_dim, train)
        self.pad = ops.Pad(((0, 0), (0, 0), (0, 512 - input_dim)))

    def construct(self, x_cmap, x_seq):
        """construct"""
        seq = self.pad(x_seq)
        x_1 = self.lstm(seq)
        x_1 = self.lm_dense(x_1)
        x_2 = self.aa_dense(x_seq)
        x_seq = self.add(x_1, x_2)
        x_seq = self.activation(x_seq)
        gx_1 = self.multi_graph_conv_1([x_seq, x_cmap])
        gx_2 = self.multi_graph_conv_2([gx_1, x_cmap])
        gx_3 = self.multi_graph_conv_3([gx_2, x_cmap])
        x = self.cat((gx_1, gx_2, gx_3))
        x = self.sum_pooling(x)
        x = self.en_dense(x)
        x = self.dropout(x)
        x = self.func_predictor(x)
        return x
