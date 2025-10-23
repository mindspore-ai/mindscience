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
"""atom update block"""

import mindspore as ms
import mindspore.mint as mint
from mindspore.common.initializer import initializer
from mindchemistry.graph.graph import AggregateEdgeToNode
from mindchemistry.utils.load_config import load_yaml_config_from_path

from .base_layers import DenseWithActivation, ResidualLayer
from .he_orthogonal import he_orthogonal_init


class AtomUpdateBlock(ms.nn.Cell):
    r"""
    Aggregate the message embeddings of the atoms

    Args:
        emb_size_atom (int): Embedding size of the atoms.
        emb_size_edge (int): Embedding size of the edges.
        n_hidden (int): Number of residual blocks.
        activation (str): Name of the activation function to use in the dense layers.
        name (str): Name of the cell. Default: "atom_update".

    Inputs:
        - **m** (Tensor) - The shape of tensor is :math:`(total\_edges, emb\_size\_edge)`.
        - **rbf** (Tensor) - The shape of tensor is :math:`(total\_edges, emb\_size\_rbf)`.
        - **id_j** (Tensor) - The shape of tensor is :math:`(total\_triplets,)`.
        - **total_atoms** (int) - Total number of atoms.
        - **idx** (int) - Index of the block.

    Outputs:
        - **x** (Tensor) - The shape of tensor is :math:`(total\_atoms, emb\_size\_atom)`.
    """

    def __init__(
            self,
            config_path,
            emb_size_atom,
            emb_size_edge,
            emb_size_rbf,
            n_hidden,
            activation=None,
            name="atom_update",
    ):
        super().__init__()
        self.name = name

        self.dense_rbf = DenseWithActivation(
            emb_size_rbf, emb_size_edge, activation=None, bias=False
        )

        self.layers = self.get_mlp(
            emb_size_edge, emb_size_atom, n_hidden, activation
        )
        self.aggregate = AggregateEdgeToNode(mode="sum")
        self.configs = load_yaml_config_from_path(config_path)
        scale_configs = self.configs.get("Scaler")
        self.scale_atom = [0,
                           scale_configs.get("AtomUpdate_1_sum"),
                           scale_configs.get("AtomUpdate_2_sum"),
                           scale_configs.get("AtomUpdate_3_sum")]

    def get_mlp(self, units_in, units, n_hidden, activation):
        dense1 = DenseWithActivation(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, n_layers=2, activation=activation)
            for i in range(n_hidden)
        ]
        mlp += res
        return ms.nn.CellList(mlp)

    def construct(self, m, rbf, id_j, total_atoms, idx):
        mlp_rbf = self.dense_rbf(rbf)
        x = mint.mul(m, mlp_rbf)
        out = mint.zeros((total_atoms, x.shape[1]))
        x = self.aggregate(x, id_j.reshape(1, -1), out)
        x = mint.mul(self.scale_atom[idx], x)

        for layer in self.layers:
            x = layer(x)
        return x


class OutputBlock(AtomUpdateBlock):
    r"""
    Combines the atom update block and subsequent final dense layer.

    Args:
        emb_size_atom (int): Embedding size of the atoms.
        emb_size_edge (int): Embedding size of the edges.
        n_hidden (int): Number of residual blocks.
        num_targets (int): Number of targets.
        activation (str): Name of the activation function to use
            in the dense layers except for the final dense layer.
        direct_forces (bool): If true directly predict forces without taking
            the gradient of the energy potential. Default: True.
        output_init (int): Kernel initializer of the final dense layer. Default: "HeOrthogonal".
        name (str): Name of the cell. Default: "output".
    Inputs:
        - **m** (Tensor) - The shape of tensor is :math:`(total\_edges, emb\_size\_edge)`.
        - **rbf** (Tensor) - The shape of tensor is :math:`(total\_edges, emb\_size\_rbf)`.
        - **id_j** (Tensor) - The shape of tensor is :math:`(total\_triplets, )`.
        - **total_atoms** (int) - Total number of atoms.
        - **idx** (int) - Index of the block.
    Outputs:
        - **x_e** (Tensor) - The shape of tensor is :math:`(total\_atoms, num\_targets)`.
        - **x_f** (Tensor) - The shape of tensor is :math:`(total\_edges, num\_targets)`.
    Raises:
        UserWarning: If the output_init is not "HeOrthogonal" or "zeros".
    """

    def __init__(
            self,
            emb_size_atom,
            emb_size_edge,
            emb_size_rbf,
            n_hidden,
            num_targets,
            activation=None,
            direct_forces=True,
            output_init="HeOrthogonal",
            name="output",
            **kwargs,
    ):

        super().__init__(
            name=name,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            n_hidden=n_hidden,
            activation=activation,
            **kwargs,
        )

        assert isinstance(output_init, str)
        self.output_init = output_init.lower()
        self.direct_forces = direct_forces

        self.seq_energy = self.get_mlp(
            emb_size_edge, emb_size_atom, n_hidden, activation
        )
        self.out_energy = DenseWithActivation(
            emb_size_atom, num_targets, bias=False, activation=None
        )

        if self.direct_forces:
            self.seq_forces = self.get_mlp(
                emb_size_edge, emb_size_edge, n_hidden, activation
            )
            self.out_forces = DenseWithActivation(
                emb_size_edge, num_targets, bias=False, activation=None
            )
            self.dense_rbf_f = DenseWithActivation(
                emb_size_rbf, emb_size_edge, activation=None, bias=False
            )
        self.aggregate = AggregateEdgeToNode(mode="sum")
        scale_configs = self.configs.get("Scaler")
        self.scale_sum = [scale_configs.get("OutBlock_0_sum"),
                          scale_configs.get("OutBlock_1_sum"),
                          scale_configs.get("OutBlock_2_sum"),
                          scale_configs.get("OutBlock_3_sum")]
        self.scale_out = [scale_configs.get("OutBlock_0_had"),
                          scale_configs.get("OutBlock_1_had"),
                          scale_configs.get("OutBlock_2_had"),
                          scale_configs.get("OutBlock_3_had")]
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of this OutputBlock"""
        if self.output_init == "heorthogonal":
            self.out_energy.reset_parameters(he_orthogonal_init)
            if self.direct_forces:
                self.out_forces.reset_parameters(he_orthogonal_init)
        elif self.output_init == "zeros":
            self.out_energy.set_data(initializer(
                "zero", self.out_energy.shape, self.out_energy.dtype))
            if self.direct_forces:
                self.out_forces.set_data(initializer(
                    "zero", self.out_forces.shape, self.out_forces.dtype))
        else:
            raise UserWarning(f"Unknown output_init: {self.output_init}")

    def construct(self, m, rbf, id_j, total_atoms, idx):

        # -------------------------------------- Energy Prediction -------------------------------------- #
        rbf_emb_e = self.dense_rbf(rbf)

        x = mint.mul(m, rbf_emb_e)
        out = mint.zeros((total_atoms, x.shape[1]))
        x_e = self.aggregate(x, id_j.reshape(1, -1), out)
        x_e = mint.mul(self.scale_sum[idx], x_e)

        for layer in self.seq_energy:
            x_e = layer(x_e)

        x_e = self.out_energy(x_e)

        # --------------------------------------- Force Prediction -------------------------------------- #
        if self.direct_forces:
            x_f = m
            for _, layer in enumerate(self.seq_forces):
                x_f = layer(x_f)

            rbf_emb_f = self.dense_rbf_f(rbf)
            x_f = mint.mul(x_f, rbf_emb_f)
            x_f = mint.mul(self.scale_out[idx], x_f)

            x_f = self.out_forces(x_f)
        else:
            x_f = 0
        return x_e, x_f
