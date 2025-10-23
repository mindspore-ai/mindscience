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
"""interaction_block"""

import math
import mindspore as ms
import mindspore.mint as mint
from mindchemistry.utils.load_config import load_yaml_config_from_path

from .atom_update_block import AtomUpdateBlock
from .base_layers import DenseWithActivation, ResidualLayer
from .efficient import EfficientInteractionBilinear
from .embedding_block import EdgeEmbedding


class InteractionBlockTripletsOnly(ms.nn.Cell):
    r"""
    Interaction block for GemNet-T/dT.

    Args:
        config_path (str): Path to the configuration file.
        emb_size_atom (int): Embedding size of the atoms.
        emb_size_edge (int): Embedding size of the edges.
        emb_size_trip (int): (Down-projected) Embedding size
            in the triplet message passing block.
        emb_size_rbf (int): Embedding size of the radial basis transformation.
        emb_size_cbf (int): Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip (int): Embedding size of the edge embeddings
            in the triplet-based message passing block after the bilinear layer.
        num_before_skip (int): Number of residual blocks before the first skip connection.
        num_after_skip (int): Number of residual blocks after the first skip connection.
        num_concat (int): Number of residual blocks after the concatenation.
        num_atom (int): Number of residual blocks in the atom embedding blocks.
        activation (str): Name of the activation function to use
            in the dense layers except for the final dense layer.
        name (str): Name of the cell. Default: "Interaction"

    Inputs:
        - **h** (Tensor): Atom embeddings. The shape of tensor is :math:`(total\_atoms, emb\_size\_atom).`
        - **m** (Tensor): Edge embeddings. The shape of tensor is :math:`(total\_edges, emb\_size\_edge).`
        - **rbf3** (Tensor): Radial basis functions. The shape of tensor is :math:`(total\_edges, emb\_size\_rbf).`
        - **rbf_w1** (Tensor): Circular basis functions.
          The shape of tensor is :math:`(total\_edges, emb\_size\_cbf, num\_spherical).`
        - **sph** (Tensor): Circular basis functions.
          The shape of tensor is :math:`(total\_edges, num\_spherical, id3\_ragged\_idx\_max+1).`
        - **id3_ragged_idx** (Tensor): Ragged index for the triplet interactions.
          The shape of tensor is :math:`(total\_triplets).`
        - **id3_ragged_idx_max** (int): Maximum index for the ragged index.
        - **id_swap** (Tensor): Swap indices for the triplet interactions.
          The shape of tensor is :math:`(total\_triplets).`
        - **id3_ba** (Tensor): Index for the triplet interactions.
          The shape of tensor is :math:`(total\_triplets).`
        - **id3_ca** (Tensor): Index for the triplet interactions.
          The shape of tensor is :math:`(total\_triplets).`
        - **rbf_h** (Tensor): Radial basis functions.
          The shape of tensor is :math:`(total\_edges, emb\_size\_rbf).`
        - **idx_s** (Tensor): Index for the edge embeddings.
          The shape of tensor is :math:`(total\_edges).`
        - **idx_t** (Tensor): Index for the edge embeddings.
          The shape of tensor is :math:`(total\_edges).`
        - **total_atoms** (int): Total number of atoms.
        - **idx** (int): Index for the triplet interactions.

    Outputs:
        - **h** (Tensor): Atom embeddings. The shape of tensor is :math:`(total\_edges, emb\_size\_atom).`
        - **m** (Tensor): Edge embeddings. The shape of tensor is :math:`(total\_edges, emb\_size\_edge).`
    """

    def __init__(
            self,
            config_path,
            emb_size_atom,
            emb_size_edge,
            emb_size_trip,
            emb_size_rbf,
            emb_size_cbf,
            emb_size_bil_trip,
            num_before_skip,
            num_after_skip,
            num_concat,
            num_atom,
            activation=None,
            name="Interaction",
    ):
        super().__init__()
        self.name = name

        block_nr = name.split("_")[-1]

        ## -------------------------------------------- Message Passing ------------------------------------------- ##
        # Dense transformation of skip connection
        self.dense_ca = DenseWithActivation(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Triplet Interaction
        self.trip_interaction = TripletInteraction(
            config_path=config_path,
            emb_size_edge=emb_size_edge,
            emb_size_trip=emb_size_trip,
            emb_size_bilinear=emb_size_bil_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            activation=activation,
            name=f"TripInteraction_{block_nr}",
        )

        ## ---------------------------------------- Update Edge Embeddings ---------------------------------------- ##
        # Residual layers before skip connection
        self.layers_before_skip = ms.nn.CellList(
            [
                ResidualLayer(
                    emb_size_edge,
                    activation=activation,
                )
                for i in range(num_before_skip)
            ]
        )

        # Residual layers after skip connection
        self.layers_after_skip = ms.nn.CellList(
            [
                ResidualLayer(
                    emb_size_edge,
                    activation=activation,
                )
                for i in range(num_after_skip)
            ]
        )

        ## ---------------------------------------- Update Atom Embeddings ---------------------------------------- ##
        self.atom_update = AtomUpdateBlock(
            config_path=config_path,
            emb_size_atom=emb_size_atom,
            emb_size_edge=emb_size_edge,
            emb_size_rbf=emb_size_rbf,
            n_hidden=num_atom,
            activation=activation,
            name=f"AtomUpdate_{block_nr}",
        )

        ## ------------------------------ Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        self.concat_layer = EdgeEmbedding(
            emb_size_atom,
            emb_size_edge,
            emb_size_edge,
            activation=activation,
        )
        self.residual_m = ms.nn.CellList(
            [
                ResidualLayer(emb_size_edge, activation=activation)
                for _ in range(num_concat)
            ]
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def construct(
            self,
            h,
            m,
            rbf3,
            rbf_w1,
            sph,
            id3_ragged_idx,
            id3_ragged_idx_max,
            id_swap,
            id3_ba,
            id3_ca,
            rbf_h,
            idx_s,
            idx_t,
            total_atoms,
            idx
    ):
        """Construct of the Interaction block."""
        # Initial transformation
        x_ca_skip = self.dense_ca(m)

        x3 = self.trip_interaction(
            m,
            rbf3,
            rbf_w1,
            sph,
            id3_ragged_idx,
            id3_ragged_idx_max,
            id_swap,
            id3_ba,
            id3_ca,
            idx,
        )

        ## ----------------------------- Merge Embeddings after Triplet Interaction ------------------------------ ##
        x = x_ca_skip + x3
        x = x * self.inv_sqrt_2

        ## ---------------------------------------- Update Edge Embeddings --------------------------------------- ##
        # Transformations before skip connection
        for _, layer in enumerate(self.layers_before_skip):
            x = layer(x)
        # Skip connection
        m = m + x
        m = m * self.inv_sqrt_2
        # Transformations after skip connection
        for _, layer in enumerate(self.layers_after_skip):
            m = layer(m)
        ## ---------------------------------------- Update Atom Embeddings --------------------------------------- ##
        h2 = self.atom_update(m, rbf_h, idx_t, total_atoms, idx)

        # Skip connection
        h = h + h2
        h = h * self.inv_sqrt_2

        ## ----------------------------- Update Edge Embeddings with Atom Embeddings ----------------------------- ##
        m2 = self.concat_layer(h, m, idx_s, idx_t)
        for _, layer in enumerate(self.residual_m):
            m2 = layer(m2)

        # Skip connection
        m = m + m2
        m = mint.mul(m, self.inv_sqrt_2)
        return h, m


class TripletInteraction(ms.nn.Cell):
    r"""
    Triplet-based message passing block.

    Args:
        config_path (str): Path to the configuration file.
        emb_size_edge (int): Embedding size of the edges.
        emb_size_trip (int): (Down-projected) Embedding size of the edge embeddings after the hadamard product with rbf.
        emb_size_bilinear (int): Embedding size of the edge embeddings after the bilinear layer.
        emb_size_rbf (int): Embedding size of the radial basis transformation.
        emb_size_cbf (int): Embedding size of the circular basis transformation (one angle).
        activation (str): Name of the activation function to use
            in the dense layers except for the final dense layer. Default: None.
        name (str): Name of the cell. Default: "TripletInteraction"

    Inputs:
        - **m** (Tensor): Edge embeddings. The shape of tensor is :math:`(total\_edges, emb\_size\_edge).`
        - **rbf3** (Tensor): Radial basis functions. The shape of tensor is :math:`(total\_edges, emb\_size\_rbf).`
        - **rbf_w1** (Tensor): Circular basis functions.
          The shape of tensor is :math:`(total\_edges, emb\_size\_cbf, num\_spherical).`
        - **sph** (Tensor): Circular basis functions.
          The shape of tensor is :math:`(total\_edges, num\_spherical, id3\_ragged\_idx\_max+1).`
        - **id3_ragged_idx** (Tensor): Ragged index for the triplet interactions.
          The shape of tensor is :math:`(total\_triplets,).`
        - **id3_ragged_idx_max** (int): Maximum index for the ragged index.
        - **id_swap** (Tensor): Swap indices for the triplet interactions.
          The shape of tensor is :math:`(total\_triplets,).`
        - **id3_ba** (Tensor): Index for the triplet interactions.
          The shape of tensor is :math:`(total\_triplets,).`
        - **id3_ca** (Tensor): Index for the triplet interactions.
          The shape of tensor is :math:`(total\_triplets,).`
        - **idx** (int): Index for the triplet interactions.

    Outputs:
        - **m** (Tensor): Edge embeddings. The shape of tensor is :math:`(total\_edges, emb\_size\_edge).`
    """

    def __init__(
            self,
            config_path,
            emb_size_edge,
            emb_size_trip,
            emb_size_bilinear,
            emb_size_rbf,
            emb_size_cbf,
            activation=None,
            name="TripletInteraction",
    ):
        super().__init__()
        self.name = name

        # Dense transformation
        self.dense_ba = DenseWithActivation(
            emb_size_edge,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        # Up projections of basis representations, bilinear layer and scaling factors
        self.mlp_rbf = DenseWithActivation(
            emb_size_rbf,
            emb_size_edge,
            activation=None,
            bias=False,
        )

        self.mlp_cbf = EfficientInteractionBilinear(
            emb_size_trip, emb_size_cbf, emb_size_bilinear
        )

        # Down and up projections
        self.down_projection = DenseWithActivation(
            emb_size_edge,
            emb_size_trip,
            activation=activation,
            bias=False,
        )
        self.up_projection_ca = DenseWithActivation(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        self.up_projection_ac = DenseWithActivation(
            emb_size_bilinear,
            emb_size_edge,
            activation=activation,
            bias=False,
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        configs = load_yaml_config_from_path(config_path)
        scale_configs = configs.get("Scaler")
        self.scale_rbf = [0,
                          scale_configs.get("TripInteraction_1_had_rbf"),
                          scale_configs.get("TripInteraction_2_had_rbf"),
                          scale_configs.get("TripInteraction_3_had_rbf")]
        self.scale_cbf = [0,
                          scale_configs.get("TripInteraction_1_sum_cbf"),
                          scale_configs.get("TripInteraction_2_sum_cbf"),
                          scale_configs.get("TripInteraction_3_sum_cbf")]

    def construct(
            self,
            m,
            rbf3,
            rbf_w1,
            sph,
            id3_ragged_idx,
            id3_ragged_idx_max,
            id_swap,
            id3_ba,
            id3_ca,
            idx,
    ):
        """Construct of the TripletInteraction block."""
        # Dense transformation
        x_ba = self.dense_ba(m)

        # Transform via radial bessel basis
        rbf_emb = self.mlp_rbf(rbf3)
        x_ba2 = mint.mul(x_ba, rbf_emb)
        x_ba = mint.mul(self.scale_rbf[idx], x_ba2)
        x_ba = self.down_projection(x_ba)

        # Transform via circular spherical basis
        x_ba = mint.index_select(x_ba, 0, id3_ba)
        # Efficient bilinear layer
        x = self.mlp_cbf(rbf_w1, sph, x_ba, id3_ca,
                         id3_ragged_idx, id3_ragged_idx_max)
        x = mint.mul(self.scale_cbf[idx], x)

        # Up project embeddings
        x_ca = self.up_projection_ca(x)
        x_ac = self.up_projection_ac(x)

        # Merge interaction of c->a and a->c
        x_ac = mint.index_select(x_ac, 0, id_swap)
        x3 = mint.add(x_ca, x_ac)
        x3 = mint.mul(x3, self.inv_sqrt_2)

        return x3
