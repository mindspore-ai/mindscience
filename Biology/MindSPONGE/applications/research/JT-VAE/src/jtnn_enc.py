# Copyright 2022 Huawei Technologies Co., Ltd
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
"""jtnn_enc"""
from collections import deque
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from .nnutils import gru
from .utils import for_concat


MAX_NB = 8


class JTNNEncoder(nn.Cell):
    """jtnnencoder"""

    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.w_z = nn.Dense(2 * hidden_size, hidden_size)
        self.w_r = nn.Dense(hidden_size, hidden_size, has_bias=False)
        self.u_r = nn.Dense(hidden_size, hidden_size)
        self.w_h = nn.Dense(2 * hidden_size, hidden_size)
        self.w = nn.Dense(2 * hidden_size, hidden_size)

    def construct(self, root_batch):
        """construct"""
        orders = []
        for root in root_batch:
            order = get_prop_order(root)
            orders.append(order)

        h = {}
        max_depth = max([len(x) for x in orders])
        padding = ops.zeros(self.hidden_size, ms.float32)

        for t in range(max_depth):
            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            cur_x = []
            cur_h_nei = []
            for node_x, node_y in prop_list:
                x, y = node_x.idx, node_y.idx
                cur_x.append(node_x.wid)

                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y:
                        continue
                    h_nei.append(h.get((z, x)))

                pad_len = MAX_NB - len(h_nei)
                h_nei.extend([padding] * pad_len)
                cur_h_nei.extend(h_nei)

            cur_x = ms.Tensor(cur_x, ms.int64)
            cur_x = self.embedding(cur_x)
            cur_h_nei = for_concat(cur_h_nei, 0)
            cur_h_nei = cur_h_nei.view(-1, MAX_NB, self.hidden_size)

            new_h = gru(cur_x, cur_h_nei, self.w_z, self.w_r, self.u_r, self.w_h)
            for i, m in enumerate(prop_list):
                x, y = m[0].idx, m[1].idx
                h[(x, y)] = new_h[i]

        root_vecs = node_aggregate(root_batch, h, self.embedding, self.w)

        tree_mess = []
        for key, value in h.items():
            tree_mess.append((key, value))

        return tree_mess, root_vecs


def get_prop_order(root):
    """get prop order"""
    queue = deque([root])
    visited = set([root.idx])
    root.set_depth(0)
    order1, order2 = [], []
    while queue:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.set_depth(x.depth + 1)
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth - 1].append((x, y))
                order2[y.depth - 1].append((y, x))
    order = order2[::-1] + order1
    return order


def node_aggregate(nodes, h, embedding, w_input):
    """node aggregate"""
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_size
    padding = ops.zeros(hidden_size, ms.float32)

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = []
        for node_y in node_x.neighbors:
            nei.append(h[(node_y.idx, node_x.idx)])
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)

    h_nei = ops.concat(h_nei, 0).view(-1, MAX_NB, hidden_size)
    sum_h_nei = h_nei.sum(axis=1)
    x_vec = ms.Tensor(x_idx, ms.int64)
    x_vec = embedding(x_vec)
    node_vec = ops.concat([x_vec, sum_h_nei], 1)
    return nn.ReLU()(w_input(node_vec))
