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
"""DimentetPlusPlus"""
import logging
import mindspore as ms
import mindspore.mint as mint
from mindspore.common.initializer import initializer
from ...graph.graph import Aggregate, AggregateEdgeToNode, AggregateNodeToGlobal
from .dimenet_utils import (
    BesselBasisLayer,
    EmbeddingBlock,
    ResidualLayer,
    swish,
)
from .glorot_orthogonal import glorot_orthogonal

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class AggregateCDVAE(Aggregate):
    r"""
    AggregateCDVAE, aggregate from triplets to edges
    Args:
        mode (str): The mode of aggregation. Default: ``sum``.
    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(total\_triplets, *)`.
        - **idx** (Tensor) - The shape of tensor is :math:`(total\_triplets,)`.
        - **out** (Tensor) - The shape of tensor is :math:`(total\_edges, *)`.
    Outputs:
        - **out** (Tensor) - The shape of tensor is :math:`(total\_edges, *)`.
    """

    def __init__(self, mode="sum"):
        super().__init__(mode=mode)

    def construct(self, x, idx, out):
        """construct"""
        return self.scatter(x, idx, out)


class InteractionPPBlock(ms.nn.Cell):
    r"""
    InteractionPPblock for DimenetPlusPlus

    Args:
        hidden_channels (int): Hidden embedding size.
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used in the basis transformation.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        num_before_skip (int): Number of residual layers in the
            interaction blocks before the skip connection.
        num_after_skip (int): Number of residual layers in the
            interaction blocks after the skip connection..
        act: (function): The activation function. Default: ``swish``.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(total\_edges,hidden\_channels)`.
        - **rbf** (Tensor) - The shape of tensor is :math:`(total\_edges, num\_radial)`.
        - **sbf** (Tensor) - The shape of tensor is :math:`(tptal_triplets, num\_radial * num\_spherical)`.
        - **idx_kj** (Tensor) - The shape of tensor is :math:`(total\_triplets,)`.
        - **idx_ji** (Tensor) - The shape of tensor is :math:`(total\_triplets,)`.

    Outputs:
        - **h** (Tensor) - The shape of tensor is :math:`(total\_edges, hidden\_channels)`.
    """

    def __init__(
            self,
            hidden_channels,
            int_emb_size,
            basis_emb_size,
            num_spherical,
            num_radial,
            num_before_skip,
            num_after_skip,
            act=swish,
    ):
        super().__init__()
        self.act = act
        self.lin_rbf1 = mint.nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = mint.nn.Linear(
            basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = mint.nn.Linear(
            num_spherical * num_radial, basis_emb_size, bias=False
        )
        self.lin_sbf2 = mint.nn.Linear(
            basis_emb_size, int_emb_size, bias=False)
        # Dense transformations of input messages.
        self.lin_kj = mint.nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = mint.nn.Linear(hidden_channels, hidden_channels)
        # Embedding projections for interaction triplets.
        self.lin_down = mint.nn.Linear(
            hidden_channels, int_emb_size, bias=False)
        self.lin_up = mint.nn.Linear(int_emb_size, hidden_channels, bias=False)
        # Residual layers before and after skip connection.
        self.layers_before_skip = ms.nn.CellList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_before_skip)
            ]
        )
        self.lin = mint.nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = ms.nn.CellList(
            [
                ResidualLayer(hidden_channels, act)
                for _ in range(num_after_skip)
            ]
        )
        self.aggregate = AggregateCDVAE(mode="sum")
        self.reset_parameters()

    def reset_parameters(self):
        """interaction block reset parameters"""
        self.lin_rbf1.weight = glorot_orthogonal(
            self.lin_rbf1.weight, scale=2.0)
        self.lin_rbf2.weight = glorot_orthogonal(
            self.lin_rbf2.weight, scale=2.0)
        self.lin_sbf1.weight = glorot_orthogonal(
            self.lin_sbf1.weight, scale=2.0)
        self.lin_sbf2.weight = glorot_orthogonal(
            self.lin_sbf2.weight, scale=2.0)
        self.lin_kj.weight = glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.set_data(initializer(
            "zero", self.lin_kj.bias.shape, self.lin_kj.bias.dtype))
        self.lin_ji.weight = glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.set_data(initializer(
            "zero", self.lin_ji.bias.shape, self.lin_ji.bias.dtype))
        self.lin_down.weight = glorot_orthogonal(
            self.lin_down.weight, scale=2.0)
        self.lin_up.weight = glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        self.lin.weight = glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.set_data(initializer(
            "zero", self.lin.bias.shape, self.lin.bias.dtype))
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def construct(self, x, rbf, sbf, idx_kj, idx_ji):
        """interaction block construct"""
        # Initial transformations.
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        # Transformation via Bessel basis.
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = mint.mul(x_kj, rbf)
        # Down-project embeddings and generate interaction triplet embeddings.
        x_kj = self.act(self.lin_down(x_kj))
        # Transform via 2D spherical basis.
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = mint.mul(mint.index_select(x_kj, 0, idx_kj), sbf)
        # Aggregate interactions and up-project embeddings.
        out = mint.zeros((x.shape[0], x_kj.shape[1]))
        x_kj = self.aggregate(x_kj, idx_ji, out)
        x_kj = self.act(self.lin_up(x_kj))
        h = mint.add(x_ji, x_kj)
        for layer in self.layers_before_skip:
            h = layer(h)
        h = mint.add(self.act(self.lin(h)), x)
        for layer in self.layers_after_skip:
            h = layer(h)
        return h


class OutputPPBlock(ms.nn.Cell):
    r"""
    OutputPPBlock for DimenetPlusPlus

    Args:
        num_radial (int): Number of radial basis functions.
        hidden_channels (int): Hidden embedding size.
        out_emb_channels (int): Embedding size used for atoms in the output block.
        out_channels (int): The size of output channels.
        num_layers (int): Number of layers.
        act: (function): The activation function. Default: ``swish``.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(total\_edges, hidden\_channels)`.
        - **rbf** (Tensor) - The shape of tensor is :math:`(total\_edges, num\_radial)`.
        - **i** (int) - Index of the OutputPPBlock
        - **total_atoms** (int) - Number of atoms.

    Outputs:
        - **x** (Tensor) - The shape of tensor is :math:`(total\_atoms, out\_channel)`.
    """

    def __init__(
            self,
            num_radial,
            hidden_channels,
            out_emb_channels,
            out_channels,
            num_layers,
            act=swish,
    ):
        super().__init__()
        self.act = act
        self.lin_rbf = mint.nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = mint.nn.Linear(
            hidden_channels, out_emb_channels, bias=True)
        self.lins = ms.nn.CellList(
            [mint.nn.Linear(out_emb_channels, out_emb_channels) for _ in range(num_layers)])
        self.lin = mint.nn.Linear(out_emb_channels, out_channels, bias=False)
        self.aggregate = AggregateEdgeToNode(mode="sum")
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rbf.weight = glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        self.lin_up.weight = glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            lin.weight = glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.set_data(initializer(
                "zero", lin.bias.shape, lin.bias.dtype))
        self.lin.weight.set_data(initializer(
            "zero", self.lin.weight.shape, self.lin.weight.dtype))

    def construct(self, x, rbf, i, total_atoms):
        x = mint.mul(self.lin_rbf(rbf), x)
        out = mint.zeros((total_atoms, x.shape[1]))
        x = self.aggregate(x, i.reshape((1, -1)), out)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class DimeNetPlusPlus(ms.nn.Cell):
    r"""
    DimenetPlusPlus Encoder.

    Args:
        num_targets (int): number of predicting targets
        hidden_channels (int): Hidden embedding size. Default: ``128``.
        latent_dim(int): Dimension of the latent parameter 'z'. Default: ``256``.
        num_blocks(int): Number of building blocks. Default: ``4``.
        int_emb_size(int): Embedding size used for interaction triplets. Default: ``64``.
        basis_emb_size (int): Embedding size used in the basis transformation. Default: ``8``.
        out_emb_channels(int): Embedding size used for atoms in the output block. Default: ``256``.
        num_spherical (int): Number of spherical harmonics. Default: ``7``.
        num_radial (int): Number of radial basis functions. Default: ``6``.
        cutoff (float): Cutoff distance for interatomic interactions. Default: ``10.0``.
        envelope_exponent (int): Shape of the smooth cutoff. Default: ``5``.
        num_before_skip (int): Number of residual layers in the
            interaction blocks before the skip connection.  Default: ``256``.
        num_after_skip (int): Number of residual layers in the
            interaction blocks after the skip connection.  Default: ``256``.
        num_output_layers (int): Number of linear layers for the output blocks.  Default: ``256``.
        readout (int): : Readout type, "mean" or "sum" Default: ``mean``.
        act: (function): The activation function. Default: ``swish``.

    Inputs:
        - **atom_types** (Tensor) - The shape of tensor is :math:`(total\_atoms,)`.
        - **dist** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.
        - **idx_kj** (Tensor) - The shape of tensor is :math:`(total\_triplets,)`.
        - **idx_ji** (Tensor) - The shape of tensor is :math:`(toal\_triplets,)`.
        - **edge_j** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.
        - **edge_i** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.
        - **batch** (Tensor) - The shape of tensor is :math:`(total\_atoms,)`.
        - **total_atoms** (int) - Number of atoms.
        - **batch_size** (int) - Batch size.
        - **sbf** (Tensor) - The shape of tensor is :math:`(tptal\_triplets, num\_radial * num\_spherical

    Outputs:
        - **energy** (Tensor) - The shape of tensor is :math:`(total\_atoms, num\_targets)`.
    """

    def __init__(
            self,
            num_targets,
            hidden_channels=128,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=10.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
            readout="mean",
            act=swish,
    ):
        super().__init__()
        out_channels = num_targets
        self.cutoff = cutoff
        self.num_blocks = num_blocks
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)
        self.output_blocks = ms.nn.CellList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self.interaction_blocks = ms.nn.CellList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )
        self.readout = readout
        self.aggregate_energy = AggregateNodeToGlobal(mode=self.readout)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def construct(self, atom_types, dist, idx_kj, idx_ji, edge_j, edge_i, batch, total_atoms, batch_size, sbf):
        """construct"""
        rbf = self.rbf(dist)
        # Embedding block.
        x = self.emb(atom_types, rbf, edge_i, edge_j)
        p = self.output_blocks[0](x, rbf, edge_i, total_atoms)
        for i in range(self.num_blocks):
            x = self.interaction_blocks[i](x, rbf, sbf, idx_kj, idx_ji)
            p += self.output_blocks[i+1](x, rbf, edge_i, total_atoms)
        if batch is None:
            if self.readout == "mean":
                energy = p.mean(dim=0)
            elif self.readout == "sum":
                energy = p.sum(dim=0)
            elif self.readout == "cat":
                energy = mint.cat((p.sum(dim=0), p.mean(dim=0)))
            else:
                raise NotImplementedError
        else:
            out = mint.zeros((batch_size, p.shape[1]))
            energy = self.aggregate_energy(p, batch, out)

        return energy
