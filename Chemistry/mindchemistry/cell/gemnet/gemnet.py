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
"""gemnet"""

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
from mindchemistry.graph.graph import AggregateEdgeToNode, AggregateEdgeToGlobal

from .layers.atom_update_block import OutputBlock
from .layers.base_layers import DenseWithActivation
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.interaction_block import InteractionBlockTripletsOnly
from .layers.radial_basis import RadialBasis
from .layers.spherical_basis import CircularBasisLayer


class GemNetT(nn.Cell):

    r"""
    GemNet-T Model for CDVAE

    Args:
        num_targets (int): Number of prediction targets.
        latent_dim(int): Dimension of the latent parameter 'z'.
        config_path (str): Path to the config file.
        num_spherical(int): Controls maximum frequency. Default: ``7``.
        num_radial (int): Controls maximum frequency. Default: ``128``.
        num_blocks (int): Number of building blocks to be stacked. Default: ``3``.
        emb_size_atom (int): Embedding size of the atoms.  Default: ``512``.
        emb_size_edge (int): Embedding size of the edges. Default: ``512``.
        emb_size_trip (int): (Down-projected) Embedding size in the triplet message passing block. Default: ``64``.
        emb_size_rbf (int): Embedding size of the radial basis transformation. Default: ``16``.
        emb_size_cbf (int): Embedding size of the circular basis transformation (one angle). Default: ``16``.
        emb_size_bil_trip (int): Embedding size of the edge embeddings in the triplet-based message
            passing block after the bilinear layer. Default: ``64``.
        num_before_skip (int): Number of residual blocks before the first skip connection. Default: ``1``.
        num_after_skip (int): Number of residual blocks after the first skip connection. Default: ``2``.
        num_concat (int): Number of residual blocks after the concatenation. Default: ``1``.
        cutoff (float): Embedding cutoff for interactomic directions in Angstrom. Default: ``6.0``.
        rbf_name (str): Name of the radial basis function. Default: ``gaussian``.
        envelope_name (str): Name of the envelope function. Default: ``polynomial``.
        envelope_exponent (int): Exponent of the envelope function. Default: ``5``.
        cbf_name (str): Name of the cosine basis function.  Default: ``spherical_harmonics``.
        output_init (str): Initialization method for the final dense layer. Default: ``HeOrthogonal``.
        activation (str): Name of the activation function. Default: ``silu``.

    Inputs:
        - **atom_types** (Tensor) - The shape of tensor is :math:`(total\_atoms)`.
        - **idx_s** (Tensor) - The shape of Tensor is :math:`(total\_edges,)`.
        - **idx_t** (Tensor) - The shape of Tensor is :math:`(total\_edges,)`.
        - **id3_ca** (Tensor) - The shape of Tensor is :math:`(total\_triplets,)`.
        - **id3_ba** (Tensor) - The shape of Tensor is :math:`(total\_triplets,)`.
        - **id3_ragged_idx** (Tensor) - The shape of Tensor is :math:`(total\_triplets,)`.
        - **id3_ragged_idx_max** (int) - Maximum number of neighbors of the edges.
        - **y_l_m** (Tensor) - The shape of Tensor is :math:`(num\_spherical, total\_triplets)`.
        - **d_st** (Tensor) - The shape of Tensor is :math:`(total\_edges,)`.
        - **v_st** (Tensor) - The shape of Tensor is :math:`(total\_edges, 3)`.
        - **id_swap** (Tensor) - The shape of Tensor is :math:`(total\_edges,)`.
        - **batch** (Tensor) - The shape of Tensor is :math:`(total\_atoms,)`.
        - **z_per_atom** (Tensor) - The shape of Tensor is :math:`(total\_atoms, latent\_dim)`.
        - **total_atoms** (int) - Total number of atoms.
        - **batch_size** (int) - Batch size.

    Outputs:
        - **res** (Tensor) - The shape of tensor is :math:`(total\_atoms, 3)`.
        - **h** (Tensor) - The shape of tensor is :math:`(total\_atoms, emb\_size\_atom)`
          if `regress_forces` is ``True``.
    """

    def __init__(
            self,
            num_targets,
            latent_dim,
            config_path,
            num_spherical=7,
            num_radial=128,
            num_blocks=3,
            emb_size_atom=512,
            emb_size_edge=512,
            emb_size_trip=64,
            emb_size_rbf=16,
            emb_size_cbf=16,
            emb_size_bil_trip=64,
            num_before_skip=1,
            num_after_skip=2,
            num_concat=1,
            num_atom=3,
            regress_forces=True,
            cutoff=6.0,
            max_neighbors=50,
            rbf_name="gaussian",
            envelope_name="polynomial",
            envelope_exponent=5,
            cbf_name="spherical_harmonics",
            output_init="HeOrthogonal",
            activation="silu",
    ):
        super().__init__()
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks

        self.cutoff = cutoff

        self.max_neighbors = max_neighbors

        self.regress_forces = regress_forces

        self.basis_functions(num_radial, cutoff, rbf_name, envelope_name,
                             envelope_exponent, num_spherical, cbf_name)

        self.share_down_projection(num_radial, emb_size_rbf, num_spherical, emb_size_cbf)

        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.atom_latent_emb = mint.nn.Linear(
            emb_size_atom + latent_dim, emb_size_atom)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        self.inter_block(config_path, emb_size_atom, emb_size_edge, emb_size_trip,
                         emb_size_rbf, emb_size_cbf, emb_size_bil_trip, num_before_skip,
                         num_after_skip, num_concat, num_atom, activation, num_blocks)
        self.out_block(config_path, emb_size_atom, emb_size_edge, emb_size_rbf, num_atom,
                       num_targets, activation, output_init, num_blocks)

        self.aggregate_sum = AggregateEdgeToGlobal(mode="sum")
        self.aggregate_mean = AggregateEdgeToNode(mode="mean")

    def basis_functions(self, num_radial, cutoff, rbf_name, envelope_name,
                        envelope_exponent, num_spherical, cbf_name):
        """
        Basis Functions
        """
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf_name=rbf_name,
            envelope_name=envelope_name,
            envelope_exponent=envelope_exponent,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf_name=rbf_name,
            envelope_name=envelope_name,
            envelope_exponent=envelope_exponent,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf_name=cbf_name,
            efficient=True,
        )

    def share_down_projection(self, num_radial, emb_size_rbf, num_spherical, emb_size_cbf):
        """Share down projection across all interaction blocks"""
        self.mlp_rbf3 = DenseWithActivation(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )
        self.mlp_rbf_h = DenseWithActivation(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = DenseWithActivation(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )

    def inter_block(self, config_path, emb_size_atom, emb_size_edge, emb_size_trip,
                    emb_size_rbf, emb_size_cbf, emb_size_bil_trip, num_before_skip,
                    num_after_skip, num_concat, num_atom, activation, num_blocks):
        """Interaction block"""
        int_blocks = []
        interaction_block = InteractionBlockTripletsOnly
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    config_path=config_path,
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    name=f"IntBlock_{i+1}",
                )
            )
        self.int_blocks = ms.nn.CellList(int_blocks)

    def out_block(self, config_path, emb_size_atom, emb_size_edge, emb_size_rbf, num_atom,
                  num_targets, activation, output_init, num_blocks):
        """Output block"""
        out_blocks = []
        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    config_path=config_path,
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    n_hidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=True,
                    name=f"OutBlock_{i}",
                )
            )

        self.out_blocks = ms.nn.CellList(out_blocks)


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def construct(self, atom_types, idx_s, idx_t, id3_ca, id3_ba,
                  id3_ragged_idx, id3_ragged_idx_max, y_l_m, d_st,
                  v_st, id_swap, batch, z_per_atom, total_atoms, batch_size):
        """
        GemNet-T Model Construct
        """
        atomic_numbers = atom_types
        rad_cbf3, cbf3 = self.cbf_basis3(d_st, y_l_m)
        rbf = self.radial_basis(d_st)
        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # Merge z and atom embedding
        if z_per_atom is not None:
            h = mint.cat([h, z_per_atom], dim=1)
            h = self.atom_latent_emb(h)
        m = self.edge_emb(h, rbf, idx_s, idx_t)
        rbf3 = self.mlp_rbf3(rbf)
        rbf_w1, sph = self.mlp_cbf3(rad_cbf3.view(1, -1, 128), cbf3, id3_ca,
                                    id3_ragged_idx, id3_ragged_idx_max)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        e_t, f_st = self.out_blocks[0](m, rbf_out, idx_t, total_atoms, 0)
        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                rbf_w1=rbf_w1,
                sph=sph,
                id3_ragged_idx=id3_ragged_idx,
                id3_ragged_idx_max=id3_ragged_idx_max,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
                total_atoms=total_atoms,
                idx=i+1
            )
            energy, force = self.out_blocks[i +
                                            1](m, rbf_out, idx_t, total_atoms, i + 1)

            f_st = mint.add(f_st, force)
            e_t = mint.add(e_t, energy)
        # always use mean aggregation
        out = mint.zeros((batch_size, e_t.shape[1]))
        e_t = self.aggregate_mean(e_t, batch.reshape(1, -1), out)

        res = None
        if self.regress_forces:
            # if predict forces, there should be only 1 energy
            assert e_t.shape[1] == 1
            # map forces in edge directions
            f_st_vec = mint.mul(f_st.view(-1, 1, 1), v_st.view(-1, 1, 3))
            out = mint.zeros(
                (total_atoms, f_st_vec.shape[1], f_st_vec.shape[2]))
            f_t = self.aggregate_sum(
                f_st_vec,
                idx_t,
                out,
            )
            f_t = f_t.squeeze(1)

            # return h for predicting atom types
            res = (h, f_t)
        else:
            res = e_t
        return res
