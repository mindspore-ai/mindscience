# Copyright 2021 Huawei Technologies Co., Ltd
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
"""model with self-interactions and gates

Exact equivariance to :math:`E(3)`

version of january 2021
"""
import math
from typing import Dict, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from   mindchemistry.e3 import o3, soft_one_hot_linspace, TensorProduct
from   mindchemistry.e3.nn import FullyConnectedNet, Gate
from   mindchemistry.e3.utils import radius_graph


def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    # Simplify the input irreducible representations to standard form.
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    # Iterate over each pair of irreps from the two input sets.
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            # Check if the output irrep is in the tensor product of the pair.
            if ir_out in ir1 * ir2:
                return True

    # Return False if no valid tensor product path is found.
    return False

def scatter_ms(src, index, dim, dim_size):
    out = src.new_zeros((dim_size, src.shape[1]))
    index = ops.unsqueeze(ms.Tensor(index, dtype=ms.int32), dim=1)
    out = ops.tensor_scatter_add(out, index, src)
    return out


class Compose(nn.Cell):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def construct(self, *input):
        x = self.first(*input)
        x = self.second(x)
        return x
    

class Convolution(nn.Cell):
    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_basis,
        radial_layers,
        radial_neurons,
        num_neighbors,
        ncon_dtype=ms.float32,
        core_mode="ncon",
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = TensorProduct(
            self.irreps_in, 
            self.irreps_node_attr, 
            self.irreps_out, 
            instructions='connect',
            ncon_dtype=ncon_dtype,
            core_mode=core_mode
        )
        self.lin1 = TensorProduct(
            self.irreps_in, 
            self.irreps_node_attr, 
            self.irreps_in,
            instructions='connect',
            ncon_dtype=ncon_dtype,
            core_mode=core_mode
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        self.tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            weight_mode='custom',
            ncon_dtype=ncon_dtype,
            core_mode=core_mode
        )
        self.fc = FullyConnectedNet(
            [number_of_basis] + radial_layers * [radial_neurons] + [self.tp.weight_numel], nn.SiLU(), dtype=ncon_dtype
        )
        self.lin2 = TensorProduct(
            irreps_mid, 
            self.irreps_node_attr, 
            self.irreps_out,
            instructions='connect',
            ncon_dtype=ncon_dtype,
            core_mode=core_mode
        )

    def construct(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded):
        weight = self.fc(edge_length_embedded)

        s = self.sc(node_input, node_attr)
 
        x = self.lin1(node_input, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)

        x = scatter_ms(edge_features, edge_dst, dim=0, dim_size=x.shape[0]).div(self.num_neighbors**0.5)

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        out = c_s * s + c_x * x
        return out


class Network(nn.Cell):
    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        irreps_out,
        irreps_node_attr,
        irreps_edge_attr,
        layers,
        max_radius,
        number_of_basis,
        radial_layers,
        radial_neurons,
        num_neighbors,
        num_nodes,
        reduce_output=True,
        ncon_dtype=ms.float32,
        core_mode="ncon",
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: nn.SiLU(),
            -1: nn.Tanh(),
        }
        act_gates = {
            1: nn.Sigmoid(),
            -1: nn.Tanh(),
        }

        self.layers = nn.CellList()

        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)]
            )
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars], 
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates], 
                irreps_gated,
                ncon_dtype=ncon_dtype,
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
                ncon_dtype=ncon_dtype,
                core_mode=core_mode,
            )
            irreps = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
                core_mode=core_mode,
            )
        )
    
    def preprocess(self, data):
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = ops.zeros(data["pos"].shape[0], ms.int32)

        edge_index, _ = radius_graph(data["pos"], self.max_radius, batch)
        edge_src = edge_index[0].tolist() # ms.Tensor(edge_index[0], ms.int32)
        edge_dst = edge_index[1].tolist() # ms.Tensor(edge_index[1], ms.int32)
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization="component")
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length, start=0.0, end=self.max_radius, number=self.number_of_basis, basis="gaussian", cutoff=True
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and "x" in data:
            assert self.irreps_in is not None
            x = data["x"]
        else:
            assert self.irreps_in is None
            x = data["pos"].new_ones((data["pos"].shape[0], 1))

        if self.input_has_node_attr and "z" in data:
            z = data["z"]
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data["pos"].new_ones((data["pos"].shape[0], 1))
        
        return batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded

    def construct(self, batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded):
        for idx, lay in enumerate(self.layers):
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)

        if self.reduce_output:
            x = scatter_ms(x, batch, dim=0).div(self.num_nodes**0.5)
        return x
