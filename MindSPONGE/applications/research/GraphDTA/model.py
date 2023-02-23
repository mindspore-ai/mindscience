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
"""docking model script."""
import sys
import numpy as np

from mindspore import nn, ops
import mindspore as ms
from mindspore_gl.nn import GCNConv2
from mindspore_gl.nn import MaxPooling

sys.path.append("..")


class GCN(nn.Cell):
    """Docking model GCN """
    def __init__(self,
                 train=True,
                 n_output=1,
                 n_filters=32,
                 embed_dim=128,
                 num_features_xd=78,
                 num_features_xt=25,
                 output_dim=128,
                 dropout=0.8,
                 atom_num=144,
                 edge_num=320,
                 batch_size=512,
                 sequence=1000):
        super().__init__()

        self.train = train
        self.edge_n = edge_num*batch_size
        self.atom_n = atom_num*batch_size
        self.n_output = n_output
        self.conv1 = GCNConv2(num_features_xd, num_features_xd)
        self.conv2 = GCNConv2(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv2(num_features_xd*2, num_features_xd*4)
        self.fc_g1 = nn.Dense(num_features_xd*4, 1024)
        self.fc_g2 = nn.Dense(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=dropout)

        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=sequence, out_channels=n_filters, kernel_size=8, pad_mode="valid")
        self.fc1_xt = nn.Dense(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Dense(2*output_dim, 1024)
        self.fc2 = nn.Dense(1024, 512)
        self.out = nn.Dense(512, self.n_output)
        self.graph_mask = ms.Tensor(np.ones(512), ms.int32)
        self.max_pooling = MaxPooling()
        self.cat = ops.Concat(axis=1)
        self.loss = nn.MSELoss()

    def construct(self, x, edge_index, target, batch, y=None):
        """docking main gcn construct"""

        x = self.conv1(x, edge_index[0], edge_index[1], self.atom_n, self.edge_n)
        x = self.relu(x)
        x = self.conv2(x, edge_index[0], edge_index[1], self.atom_n, self.edge_n)
        x = self.relu(x)

        x = self.conv3(x, edge_index[0], edge_index[1], self.atom_n, self.edge_n)
        x = self.relu(x)
        x = self.max_pooling(x, edge_index[0], edge_index[1], self.atom_n, self.edge_n, batch, batch, self.graph_mask)

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)

        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = self.cat((x, xt))
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        if self.train:
            label = y.view(-1, 1)
            loss = self.loss(out, label)
        else:
            loss = 0
        return loss, out
