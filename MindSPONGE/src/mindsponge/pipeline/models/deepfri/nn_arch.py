# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Predictor"""
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops

from .module.layers import LstmLayer, MultiGraphConv, SumPooling, FuncPredictor


class Predictor(nn.Cell):
    """
        Class for loading trained models and computing GO/EC predictions and class activation maps (CAMs).
    """

    def __init__(self, model_prefix, config, gcn=True):
        super(Predictor, self).__init__()
        self.model_prefix = model_prefix
        self.config = config
        self.gcn = gcn

        # load parameters
        self.go_names = mnp.asarray(self.config.gonames)
        self.go_terms = mnp.asarray(self.config.goterms)
        self.thresh = 0.1 * mnp.ones(len(self.go_terms))
        # load model
        # init params
        self.chain2path = {}
        self.y_hat = mnp.zeros((1, 1))
        self.test_prot_list = []
        self.goidx2chains = {}
        self.prot2goterms = {}
        self.data = {}
        self.lstm = LstmLayer(512)
        self.lm_dense = nn.Dense(1024, 1024, has_bias=True)
        self.aa_dense = nn.Dense(self.config.input_dim, 1024, has_bias=False)
        self.add = ops.Add()
        self.activation = nn.ReLU()
        train = False
        self.multi_graph_conv_1 = MultiGraphConv(1024, self.config.gc_dims[0], activation='relu', train=train)
        self.multi_graph_conv_2 = MultiGraphConv(self.config.gc_dims[0], self.config.gc_dims[1], activation='relu',
                                                 train=train)
        self.multi_graph_conv_3 = MultiGraphConv(self.config.gc_dims[1], self.config.gc_dims[2], activation='relu',
                                                 train=train)
        self.cat = ops.Concat(-1)
        self.sum_pooling = SumPooling(1)
        self.en_dense = nn.Dense((self.config.gc_dims[0] + self.config.gc_dims[1] + self.config.gc_dims[2]),
                                 self.config.fc_dims, has_bias=True, activation='relu')
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.func_predictor = FuncPredictor(self.config.fc_dims, self.config.output_dim, train)
        self.pad = ops.Pad(((0, 0), (0, 0), (0, 512 - self.config.input_dim)))

    def predict(self, adj, seq_1hot):
        """predict"""
        if self.gcn:
            seq_0 = self.pad(seq_1hot)
            x_1 = self.lstm(seq_0)
            x_1 = self.lm_dense(x_1)
            x_2 = self.aa_dense(seq_1hot)
            x_seq = self.add(x_1, x_2)
            x_seq = self.activation(x_seq)
            gx_1 = self.multi_graph_conv_1([x_seq, adj])
            gx_2 = self.multi_graph_conv_2([gx_1, adj])
            gx_3 = self.multi_graph_conv_3([gx_2, adj])
            x = self.cat((gx_1, gx_2, gx_3))
            x = self.sum_pooling(x)
            x = self.en_dense(x)
            x = self.dropout(x)
            y = self.func_predictor(x)[:, :, 0].reshape(-1)

            return y[135]
        return 0
