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
from mindchemistry.graph.graph import AggregateEdgeToNode
from mindchemistry.e3.o3 import TensorProduct, Irreps, Linear, FullyConnectedTensorProduct
from mindchemistry.e3.nn import FullyConnectedNet

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

        # self.lin1 = Linear(self.irreps_node_input, self.irreps_node_input, dtype=dtype, ncon_dtype=ncon_dtype)
        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = TensorProduct(self.irreps_node_input,
                           self.irreps_edge_attr,
                           #    self.irreps_node_output,
                           irreps_mid,
                           #    'merge',
                           instructions,
                           weight_mode='custom',
                           dtype=dtype,
                           ncon_dtype=ncon_dtype)

        self.fc = FullyConnectedNet([self.irreps_edge_scalars.num_irreps] + invariant_layers * [invariant_neurons] +
                                    [tp.weight_numel], {
                                        "ssp": shift_softplus,
                                        "silu": ops.silu,
                                    }.get(nonlin_scalars.get("e", None), None), dtype=dtype)

        self.tp = tp
        self.scatter = AggregateEdgeToNode(dim=1)

        # self.lin2 = Linear(tp.irreps_out.simplify(), self.irreps_node_output, dtype=dtype, ncon_dtype=ncon_dtype)
        self.lin2 = FullyConnectedTensorProduct(tp.irreps_out.simplify(), self.irreps_node_attr, self.irreps_node_output)

        self.sc = None
        if self.use_sc:
            # self.sc = TensorProduct(self.irreps_node_input,
            #                         self.irreps_node_attr,
            #                         self.irreps_node_output,
            #                         'connect',
            #                         dtype=dtype,
            #                         ncon_dtype=ncon_dtype)
            self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

    def construct(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars):
        """Evaluate interaction Block with resnet"""
        weight = self.fc(edge_scalars)

        # import mindspore as ms
        # import numpy as np
        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/conv_weight.npy"))
        # print('fc', (torch_res - weight).abs().max())

        # node_features = self.lin1(node_input)
        node_features = self.lin1(node_input, node_attr)

        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/conv_lin1.npy"))
        # print('lin1', (torch_res - node_features).abs().max())

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)

        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/conv_tp.npy"))
        # print('tp', (torch_res - edge_features).abs().max())


        node_features = self.scatter(edge_attr=edge_features, edge_index=[edge_src, edge_dst],
                                     dim_size=node_input.shape[0])

        if self.avg_num_neighbors is not None:
            node_features = node_features.div(self.avg_num_neighbors**0.5)

        # node_features = self.lin2(node_features)
        node_features = self.lin2(node_features, node_attr)

        # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/conv_lin2.npy"))
        # print('lin2', (torch_res - node_features).abs().max())

        if self.sc is not None:
            sc = self.sc(node_input, node_attr)

            # torch_res = ms.Tensor(np.load("/data/zmmVol2/wyh/0415/BETE-NET/BETE_NET_MS/notebooks_ms/torch_res/conv_sc.npy"))
            # print('sc', (torch_res - sc).abs().max())

            # node_features = node_features + sc
        
        import math
        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        node_features = c_s * sc + c_x * node_features

        return node_features
