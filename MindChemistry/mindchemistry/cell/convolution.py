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
"""convolution"""
from mindspore import nn, ops, float32
from ..e3.o3 import TensorProduct, Irreps, Linear
from ..e3.nn import FullyConnectedNet, Scatter

softplus = ops.Softplus()


def shift_softplus(x):
    return softplus(x) - 0.6931471805599453


def silu(x):
    return x * ops.sigmoid(x)


class Convolution(nn.Cell):
    r"""
    InteractionBlock.

    Args:
        irreps_node_input: Input Features, default = None
        irreps_node_attr: Nodes attribute irreps
        irreps_node_output: Output irreps, in our case typically a single scalar
        irreps_edge_attr: Edge attribute irreps
        invariant_layers: Number of invariant layers, default = 1
        invariant_neurons: Number of hidden neurons in invariant function, default = 8
        avg_num_neighbors: Number of neighbors to divide by, default None => no normalization.
        use_sc(bool): use self-connection or not
    """

    def __init__(self,
                 irreps_node_input,
                 irreps_node_attr,
                 irreps_node_output,
                 irreps_edge_attr,
                 irreps_edge_scalars,
                 invariant_layers=1,
                 invariant_neurons=8,
                 avg_num_neighbors=None,
                 use_sc=True,
                 nonlin_scalars=None,
                 dtype=float32,
                 ncon_dtype=float32):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc

        self.irreps_node_input = Irreps(irreps_node_input)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_node_output = Irreps(irreps_node_output)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_edge_scalars = Irreps([(irreps_edge_scalars.num_irreps, (0, 1))])

        self.lin1 = Linear(self.irreps_node_input, self.irreps_node_input, dtype=dtype)

        tp = TensorProduct(self.irreps_node_input,
                           self.irreps_edge_attr,
                           self.irreps_node_output,
                           'merge',
                           weight_mode='custom',
                           dtype=dtype,
                           ncon_dtype=ncon_dtype)

        self.fc = FullyConnectedNet([self.irreps_edge_scalars.num_irreps] + invariant_layers * [invariant_neurons] +
                                    [tp.weight_numel], {
                                        "ssp": shift_softplus,
                                        "silu": ops.silu,
                                    }.get(nonlin_scalars.get("e", None), None), dtype=dtype)

        self.tp = tp
        self.scatter = Scatter()

        self.lin2 = Linear(tp.irreps_out.simplify(), self.irreps_node_output, dtype=dtype)

        self.sc = None
        if self.use_sc:
            self.sc = TensorProduct(self.irreps_node_input,
                                    self.irreps_node_attr,
                                    self.irreps_node_output,
                                    'connect',
                                    dtype=dtype,
                                    ncon_dtype=ncon_dtype)

    def construct(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars):
        """Evaluate interaction Block with resnet"""
        weight = self.fc(edge_scalars)

        node_features = self.lin1(node_input)

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)

        node_features = self.scatter(edge_features, edge_dst, dim_size=node_input.shape[0])

        if self.avg_num_neighbors is not None:
            node_features = node_features.div(self.avg_num_neighbors**0.5)

        node_features = self.lin2(node_features)

        if self.sc is not None:
            sc = self.sc(node_input, node_attr)
            node_features = node_features + sc

        return node_features
