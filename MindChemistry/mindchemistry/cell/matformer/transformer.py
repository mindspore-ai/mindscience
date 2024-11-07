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
"""transformer file"""
import mindspore as ms
from mindspore import ops, Tensor, nn
from mindchemistry.graph.graph import LiftNodeToEdge, AggregateEdgeToNode
from mindchemistry.graph.normlization import BatchNormMask


class Silu(nn.Cell):
    """Silu"""

    def __init__(self):
        """init"""
        super(Silu, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        """construct"""
        return ops.mul(x, self.sigmoid(x))


class MatformerConv(nn.Cell):
    """MatformerConv"""

    def __init__(
            self,
            in_channels=None,
            out_channels=None,
            heads=1,
            concat=True,
            beta=False,
            edge_dim=None,
            bias=True,
            root_weight=True,
            use_fp16=False
    ):
        """init"""
        super(MatformerConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.edge_dim = edge_dim
        self.global_dtype = ms.float32
        if use_fp16:
            self.global_dtype = ms.float16

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = nn.Dense(in_channels[0], heads * out_channels).to_float(self.global_dtype)
        self.lin_query = nn.Dense(in_channels[1], heads * out_channels).to_float(self.global_dtype)
        self.lin_value = nn.Dense(in_channels[0], heads * out_channels).to_float(self.global_dtype)

        if edge_dim is not None:
            self.lin_edge = nn.Dense(edge_dim, heads * out_channels, has_bias=False).to_float(self.global_dtype)

        if concat:
            self.lin_concate = nn.Dense(heads * out_channels, out_channels).to_float(self.global_dtype)

        self.lin_skip = nn.Dense(in_channels[1], out_channels, has_bias=bias).to_float(self.global_dtype)
        if self.beta:
            self.lin_beta = nn.Dense(3 * out_channels, 1, has_bias=False).to_float(self.global_dtype)

        self.lin_msg_update = nn.Dense(out_channels * 3, out_channels * 3).to_float(self.global_dtype)

        self.msg_layer = nn.SequentialCell(nn.Dense(out_channels * 3, out_channels).to_float(self.global_dtype),
                                           nn.LayerNorm((out_channels,), epsilon=0.00001).to_float(self.global_dtype))

        self.bn = BatchNormMask(out_channels).to_float(ms.float32)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm((out_channels * 3,), epsilon=0.00001).to_float(ms.float32)
        self.silu = Silu()
        self.lift_to_edge_i = LiftNodeToEdge(dim=1)
        self.lift_to_edge_j = LiftNodeToEdge(dim=0)
        self.aggregate_to_node = AggregateEdgeToNode(mode="add", dim=1)

        self.layer_norm_after_lin_msg_update = nn.LayerNorm((out_channels * 3,), epsilon=0.00001).to_float(ms.float32)
        self.gelu = nn.GELU()

    def construct(self, x, edge_index, edge_attr, node_mask, edge_mask, node_num):
        """construct"""
        h_num, c_num = self.heads, self.out_channels
        if isinstance(x, Tensor):
            x = (x, x)
        query = self.lin_query(x[1]).view(-1, h_num, c_num)
        key = self.lin_key(x[0]).view(-1, h_num, c_num)
        value = self.lin_value(x[0]).view(-1, h_num, c_num)
        out = self.propagate(edge_index, edge_mask, query, key, value, edge_attr)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
            out = self.lin_concate(out)
        else:
            out = ops.mean(out, axis=1)

        node_mask = ops.reshape(node_mask, (-1, 1))
        out = self.bn(out, node_mask, node_num)
        out = self.silu(out)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            x_r = ops.mul(x_r, node_mask).astype(ms.float32)
            if self.beta:
                beta = self.lin_beta(ops.cat([out, x_r, out - x_r], axis=-1))
                beta = self.sigmoid(beta)
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out

    def message(self, query_i, key_i, key_j, value_j, value_i,
                edge_attr) -> Tensor:
        """message"""
        if self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)

        query_i = ops.cat((query_i, query_i, query_i), axis=-1)
        key_j = ops.cat((key_i, key_j, edge_attr), axis=-1)
        alpha = ops.div(ops.mul(query_i.astype(ms.float32), key_j.astype(ms.float32)),
                        ops.sqrt(ms.Tensor(self.out_channels * 3, ms.float32)))
        out = ops.cat((value_i, value_j, edge_attr), axis=-1)

        out_norm = self.gelu(self.layer_norm_after_lin_msg_update(self.lin_msg_update(out)))
        out = out_norm * self.sigmoid(self.layer_norm(alpha.view(-1, self.heads, 3 * self.out_channels)))
        out = self.msg_layer(out)
        return out

    def propagate(self, edge_index, trans_scatter_mask, query, key, value, edge_attr):
        """propagate"""
        query_i = self.lift_to_edge_i(query, edge_index)
        key_i = self.lift_to_edge_i(key, edge_index)
        value_i = self.lift_to_edge_i(value, edge_index)
        key_j = self.lift_to_edge_j(key, edge_index)
        value_j = self.lift_to_edge_j(value, edge_index)
        out = self.message(query_i, key_i, key_j, value_j, value_i, edge_attr)
        out = self.aggregate_to_node(out, edge_index, dim_size=query.shape[0], mask=trans_scatter_mask)
        return out

    def __repr__(self) -> str:
        """__repr__"""
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
