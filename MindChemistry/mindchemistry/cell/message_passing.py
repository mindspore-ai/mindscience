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

from mindspore import nn, ops, float32

from ..e3.o3 import Irreps
from ..e3.nn import Gate, NormActivation
from .convolution import Convolution, shift_softplus

acts = {
    "abs": ops.abs,
    "tanh": ops.tanh,
    "ssp": shift_softplus,
    "silu": ops.silu,
}


class Compose(nn.Cell):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def construct(self, *input):
        x = self.first(*input)
        x = self.second(x)
        return x


class MessagePassing(nn.Cell):

    def __init__(
            self,
            irreps_node_input,
            irreps_node_attr,
            irreps_node_hidden,
            irreps_node_output,
            irreps_edge_attr,
            irreps_edge_scalars,
            convolution_kwargs={},
            num_layers=3,
            resnet=False,
            nonlin_type="gate",
            nonlin_scalars={"e": "ssp", "o": "tanh"},
            nonlin_gates={"e": "ssp", "o": "abs"},
            dtype=float32
    ):
        super().__init__()
        if not nonlin_type in ("gate", "norm"):
            raise ValueError(f"Unexpected nonlin_type {nonlin_type}.")

        nonlin_scalars = {
            1: nonlin_scalars["e"],
            -1: nonlin_scalars["o"],
        }
        nonlin_gates = {
            1: nonlin_gates["e"],
            -1: nonlin_gates["o"],
        }

        self.irreps_node_input = Irreps(irreps_node_input)
        self.irreps_node_hidden = Irreps(irreps_node_hidden)
        self.irreps_node_output = Irreps(irreps_node_output)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_edge_scalars = Irreps(irreps_edge_scalars)

        irreps_node = self.irreps_node_input
        irreps_prev = irreps_node
        self.layers = nn.CellList()
        self.resnets = []

        for _ in range(num_layers):
            tmp_irreps = irreps_node * self.irreps_edge_attr

            irreps_scalars = Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l == 0 and ir in tmp_irreps
                ]
            ).simplify()
            irreps_gated = Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l > 0 and ir in tmp_irreps
                ]
            )

            if nonlin_type == "gate":
                ir = "0e" if Irreps("0e") in tmp_irreps else "0o"
                irreps_gates = Irreps([(mul, ir)
                                       for mul, _ in irreps_gated]).simplify()

                nonlinear = Gate(
                    irreps_scalars,
                    [acts[nonlin_scalars[ir.p]] for _, ir in irreps_scalars],
                    irreps_gates,
                    [acts[nonlin_gates[ir.p]] for _, ir in irreps_gates],
                    irreps_gated,
                    dtype=dtype
                )

                conv_irreps_out = nonlinear.irreps_in
            else:
                conv_irreps_out = (irreps_scalars + irreps_gated).simplify()

                nonlinear = NormActivation(
                    irreps_in=conv_irreps_out,
                    act=acts[nonlin_scalars[1]],
                    normalize=True,
                    epsilon=1e-8,
                    bias=False,
                    dtype=dtype
                )

            conv = Convolution(
                irreps_node_input=irreps_node,
                irreps_node_attr=self.irreps_node_attr,
                irreps_node_output=conv_irreps_out,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_edge_scalars=self.irreps_edge_scalars,
                **convolution_kwargs,
                dtype=dtype
            )
            irreps_node = nonlinear.irreps_out

            self.layers.append(Compose(conv, nonlinear))

            if irreps_prev == irreps_node and resnet:
                self.resnets.append(True)
            else:
                self.resnets.append(False)
            irreps_prev = irreps_node

    def construct(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars):
        layer_in = node_input
        for i in range(len(self.layers)):
            layer_out = self.layers[i](
                layer_in, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)

            if self.resnets[i]:
                layer_in = layer_out + layer_in
            else:
                layer_in = layer_out

        return layer_in
