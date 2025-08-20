# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: disable-all
from mindspore import nn, ops
import numpy as np
from .layer import fnn, attention, Conv1D


def dyn_input_feature_layer(x):
    return ops.cat((x, ops.cos(np.pi * x), ops.sin(np.pi * x), ops.cos(2 * np.pi * x), ops.sin(2 * np.pi * x)), axis=-1)


class three_bus_PN(nn.Cell):
    def __init__(self,
                 dynamic,
                 algebraic,
                 use_input_layer=None,
                 stacked=False):
        super().__init__()
        self.stacked = stacked
        self.dim = 4
        self.num_IRK_stages = dynamic.num_IRK_stages
        if stacked:
            num_layer = self.dim
        else:
            num_layer = 1
        dyn_in_transform = dyn_input_feature_layer if use_input_layer else None
        dyn_out_transform = None
        alg_out_transform = ops.Softplus()
        alg_in_transform = None
        if dynamic.type == "fnn":
            self.Y = nn.CellList([
                fnn(
                    dynamic.layer_size,
                    dynamic.activation,
                    dynamic.initializer,
                    dropout_rate=dynamic.dropout_rate,
                    batch_normalization=dynamic.batch_normalization,
                    layer_normalization=dynamic.layer_normalization,
                    input_transform=dyn_in_transform,
                    output_transform=dyn_out_transform
                ) for _ in range(num_layer)
            ])

        elif dynamic.type == "attention":
            self.Y = nn.CellList([
                attention(
                    dynamic.layer_size,
                    dynamic.activation,
                    dynamic.initializer,
                    dropout_rate=dynamic.dropout_rate,
                    batch_normalization=dynamic.batch_normalization,
                    layer_normalization=dynamic.layer_normalization,
                    input_transform=dyn_in_transform,
                    output_transform=dyn_out_transform
                ) for _ in range(num_layer)
            ])

        elif dynamic.type == "Conv1D":
            self.Y = nn.CellList([
                Conv1D(
                    dynamic.layer_size,
                    dynamic.activation,
                    dropout_rate=dynamic.dropout_rate,
                    batch_normalization=dynamic.batch_normalization,
                    layer_normalization=dynamic.layer_normalization,
                    input_transform=dyn_in_transform,
                    output_transform=dyn_out_transform
                ) for _ in range(num_layer)
            ])
        else:
            raise ValueError(f"{dynamic.type} type on NN not implemented")

        if algebraic.type == "fnn":
            self.Z = fnn(
                algebraic.layer_size,
                algebraic.activation,
                algebraic.initializer,
                dropout_rate=algebraic.dropout_rate,
                batch_normalization=algebraic.batch_normalization,
                layer_normalization=algebraic.layer_normalization,
                input_transform=alg_in_transform,
                output_transform=alg_out_transform
            )
        elif algebraic.type == "attention":
            self.Z = attention(
                algebraic.layer_size,
                algebraic.activation,
                algebraic.initializer,
                dropout_rate=algebraic.dropout_rate,
                batch_normalization=algebraic.batch_normalization,
                layer_normalization=algebraic.layer_normalization,
                input_transform=alg_in_transform,
                output_transform=alg_out_transform
            )
        elif algebraic.type == "Conv1D":
            self.Z = Conv1D(
                algebraic.layer_size,
                algebraic.activation,
                dropout_rate=algebraic.dropout_rate,
                batch_normalization=algebraic.batch_normalization,
                layer_normalization=algebraic.layer_normalization,
                input_transform=alg_in_transform,
                output_transform=alg_out_transform
            )
        else:
            raise ValueError(f"{algebraic.type} type on NN not implemented")

    def construct(self, inputs):
        if self.stacked:
            Y0 = self.Y[0](inputs)
            Y1 = self.Y[1](inputs)
            Y2 = self.Y[2](inputs)
            Y3 = self.Y[3](inputs)
        else:
            dim_out = self.num_IRK_stages + 1
            Y = inputs
            for layer in self.Y:
                Y = layer(Y)
            Y0 = Y[..., :dim_out]
            Y1 = Y[..., dim_out:2 * dim_out]
            Y2 = Y[..., 2 * dim_out:3 * dim_out]
            Y3 = Y[..., 3 * dim_out:4 * dim_out]
        Z = self.Z(inputs)
        return Y0, Y1, Y2, Y3, Z
