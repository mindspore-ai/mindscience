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

from mindspore import nn, ops, float32, int32, Tensor
import numpy as np

from ..e3.o3 import Irreps, SphericalHarmonics, TensorProduct, Linear
from ..e3.nn import OneHot, Scatter
from ..e3.utils import radius_graph
from .message_passing import MessagePassing
from .embedding import RadialEdgeEmbedding


class AtomwiseLinear(nn.Cell):

    def __init__(self, irreps_in, irreps_out, dtype=float32):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.linear = Linear(self.irreps_in, self.irreps_out, dtype=dtype)

    def construct(self, node_input):
        return self.linear(node_input)

    def __repr__(self):
        return self.linear.__repr__()


class EnergyNet(nn.Cell):

    def __init__(
            self,
            irreps_embedding_out,
            irreps_conv_out='16x0e',
            chemical_embedding_irreps_out='64x0e',
            r_max=4.0,
            num_layers=3,
            num_type=4,
            num_basis=8,
            cutoff_p=6,
            hidden_mul=50,
            lmax=2,
            pred_force=False,
            dtype=float32
    ):
        super().__init__()
        self.r_max = r_max
        self.irreps_conv_out = Irreps(irreps_conv_out)
        self.pred_force = pred_force
        self.irreps_embedding_out = Irreps(irreps_embedding_out)
        if pred_force:
            self.irreps_embedding_out += Irreps([(self.irreps_embedding_out.data[0].mul, (1, -1))])

        irreps_node_hidden = Irreps([(hidden_mul, (l, p))
                                     for l in range(lmax + 1) for p in [-1, 1]])

        self.one_hot = OneHot(num_type, dtype=dtype)
        self.sh = SphericalHarmonics(range(lmax + 1), True, normalization="component", dtype=dtype)
        self.radial_embedding = RadialEdgeEmbedding(r_max, num_basis, cutoff_p, dtype=dtype)

        irreps_output = Irreps(chemical_embedding_irreps_out)
        self.lin_input = AtomwiseLinear(self.one_hot.irreps_output, irreps_output, dtype=dtype)

        irreps_edge_scalars = self.radial_embedding.irreps_out

        irrep_node_features = irreps_output

        self.mp = MessagePassing(
            irreps_node_input=irrep_node_features,
            irreps_node_attr=self.one_hot.irreps_output,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=self.irreps_conv_out,
            irreps_edge_attr=self.sh.irreps_out,
            irreps_edge_scalars=irreps_edge_scalars,
            num_layers=num_layers,
            resnet=False,
            convolution_kwargs={'invariant_layers': 3, 'invariant_neurons': 64, 'avg_num_neighbors': 9,
                                'nonlin_scalars': {"e": "silu"}},
            nonlin_scalars={"e": "silu", "o": "tanh"},
            nonlin_gates={"e": "silu", "o": "tanh"},
            dtype=dtype,
        )
        self.lin1 = AtomwiseLinear(self.irreps_conv_out, self.irreps_embedding_out, dtype=dtype)

        irreps_out = '1x0e+1x1o' if pred_force else '1x0e'

        self.lin2 = AtomwiseLinear(self.irreps_embedding_out, irreps_out, dtype=dtype)

        self.scatter = Scatter()

    def preprocess(self, data):
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=int32)

        edge_index = radius_graph(
            data["pos"], self.r_max, batch, max_num_neighbors=len(data["pos"]) - 1)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        return batch, edge_src, edge_dst

    def construct(self, batch, atom_type, atom_pos, edge_src, edge_dst, batch_size):
        edge_vec = atom_pos[edge_dst] - atom_pos[edge_src]
        node_inputs = self.one_hot(atom_type)
        node_attr = node_inputs.copy()
        edge_attr = self.sh(edge_vec)

        edge_length = edge_vec.norm(None, 1)
        edge_length_embedding = self.radial_embedding(edge_length)

        node_features = self.lin_input(node_inputs)
        node_features = self.mp(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
        node_features = self.lin1(node_features)
        node_features = self.lin2(node_features)

        if self.pred_force:
            energy = self.scatter(node_features[:, :1], batch, dim_size=batch_size)
            forces = node_features[:, 1:]
            return energy, forces

        energy = self.scatter(node_features, batch, dim_size=batch_size)
        return energy
