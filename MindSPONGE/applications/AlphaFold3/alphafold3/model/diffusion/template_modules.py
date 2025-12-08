# Copyright 2024 DeepMind Technologies Limited
# Copyright (C) 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications by Huawei Technologies Co., Ltd: Adapt to run by MindSpore on Ascend

"""template modules"""
from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, ops, Tensor, mint

from alphafold3.model import base_config
from alphafold3.constants import residue_names
from alphafold3.utils import geometry
from alphafold3.model import protein_data_processing
from alphafold3.model.components import base_modules as bm
from alphafold3.model.diffusion import modules
from alphafold3.model.scoring import scoring


@dataclass
class DistogramFeaturesConfig(base_config.BaseConfig):
    # The left edge of the first bin.
    min_bin: float = 3.25
    # The left edge of the final bin. The final bin catches everything larger than
    # `max_bin`.
    max_bin: float = 50.75
    # The number of bins in the distogram.
    num_bins: int = 39


def dgram_from_positions(positions, config, dtype=ms.float32):
    """Compute distogram from amino acid positions.

    Args:
        positions: (num_res, 3) Position coordinates.
        config: Distogram bin configuration.

    Returns:
        Distogram with the specified number of bins.
    """
    lower_breaks = mint.linspace(
        config.min_bin, config.max_bin, config.num_bins)
    lower_breaks = mint.square(lower_breaks)
    upper_breaks = mint.concat(
        [lower_breaks[1:], Tensor([1e8], dtype=ms.float32)], dim=-1)
    dist2 = mint.sum(mint.square(ops.expand_dims(positions, axis=-2)
                                 - ops.expand_dims(positions, axis=-3)), dim=-1, keepdim=True)
    dgram = (dist2 > lower_breaks).astype(ms.float32) * \
        (dist2 < upper_breaks).astype(ms.float32)
    return dgram


def slice_index(x, idx):
    """Slice index."""
    return ops.gather_d(x, 1, idx.reshape(-1, 1)).squeeze()


def make_backbone_rigid(positions, mask, group_indices,):
    """Make backbone Rigid3Array and mask.

    Args:
        positions: (num_res, num_atoms) of atom positions as Vec3Array.
        mask: (num_res, num_atoms) for atom mask.
        group_indices: (num_res, num_group, 3) for atom indices forming groups.

    Returns:
        tuple of backbone Rigid3Array and mask (num_res,).
    """
    backbone_indices = group_indices[:, 0]

    # main backbone frames differ in sidechain frame convention.
    # for sidechain it's (C, CA, N), for backbone it's (N, CA, C)
    # Hence using c, b, a, each of shape (num_res,).
    c, b, a = [backbone_indices[..., i] for i in range(3)]

    rigid_mask = slice_index(mask, a) * \
        slice_index(mask, b) * slice_index(mask, c)
    frame_positions = []
    for indices in [a, b, c]:
        frame_positions.append(geometry.vector.tree_map(
            lambda x, idx=indices: slice_index(x, idx), positions
        ))
    rotation = geometry.Rot3Array.from_two_vectors(
        frame_positions[2] - frame_positions[1],
        frame_positions[0] - frame_positions[1],
    )
    rigid = geometry.Rigid3Array(rotation, frame_positions[1])
    return rigid, rigid_mask


class TemplateEmbedding(nn.Cell):
    """
    Embed a set of templates.

    Args:
        config (TemplateEmbedding.Config): Configuration for the template embedding.
        global_config (base_config.BaseConfig): Global configuration for the model.
        num_templates (int): Number of templates to process.
        normalized_shape (tuple): Shape of the normalized input tensor.
        num_atoms (int): Number of atoms per residue. Default: ``24``.

    Inputs:
        - **query_embedding** (Tensor) - Query tensor of shape [num_res, num_res, num_channel].
        - **templates** (Templates) - Object containing template data.
        - **padding_mask_2d** (Tensor) - Pair mask for attention operations of shape [num_res, num_res].
        - **multichain_mask_2d** (Tensor) - Pair mask for multichain operations of shape [num_res, num_res].
        - **key** (int) - Random key generator.

    Outputs:
        - **embedding** (Tensor) - Output embedding tensor of shape [num_res, num_res, num_channels].
    """
    @dataclass
    class Config(base_config.BaseConfig):
        num_channels: int = 64
        template_stack: modules.PairFormerIteration.Config = base_config.autocreate(
            num_layer=2,
            pair_transition=base_config.autocreate(num_intermediate_factor=2),
        )
        dgram_features: DistogramFeaturesConfig = base_config.autocreate()

    def __init__(self, config, global_config, num_templates, normalized_shape, num_atoms=24, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.num_residues = normalized_shape[0]
        self.num_templates = num_templates
        self.query_num_channels = normalized_shape[2]
        self.num_atoms = num_atoms
        self.template_embedder = SingleTemplateEmbedding(
            self.config, self.global_config, normalized_shape, dtype=dtype)
        self.output_linear = bm.CustomDense(
            self.config.num_channels, self.query_num_channels, ndim=3, dtype=dtype)
        self.output_linear.weight = bm.custom_initializer(
            'relu', (self.config.num_channels, self.query_num_channels), dtype=dtype)

    def construct(self, query_embedding, templates, padding_mask_2d,
                  multichain_mask_2d, key):
        """Generate an embedding for a set of templates.

        Args:
        query_embedding: [num_res, num_res, num_channel] a query tensor that will
            be used to attend over the templates to remove the num_templates
            dimension.
        templates: A 'Templates' object.
        padding_mask_2d: [num_res, num_res] Pair mask for attention operations.
        multichain_mask_2d: [num_res, num_res] Pair mask for multichain.
        key: random key generator.

        Returns:
        An embedding of size [num_res, num_res, num_channels]
        """
        subkeys = mint.arange(key, key + self.num_templates, 1)
        summed_template_embeddings = mint.zeros(
            (self.num_residues, self.num_residues,
             self.config.num_channels), dtype=query_embedding.dtype
        )

        def scan_fn(carry, x):
            templates, key = x
            embedding = self.template_embedder(
                query_embedding,
                templates,
                padding_mask_2d,
                multichain_mask_2d,
                key,
            )
            return carry + embedding
        for i, subkey in enumerate(subkeys):
            summed_template_embeddings = scan_fn(
                summed_template_embeddings, (templates[i], subkey))
        embedding = summed_template_embeddings / (1e-7 + self.num_templates)
        embedding = mint.nn.functional.relu(embedding)
        embedding = self.output_linear(embedding)
        return embedding


class SingleTemplateEmbedding(nn.Cell):
    """
    Embed a single template.

    Args:
        config: Configuration object containing model parameters.
        global_config: Global configuration object.
        normalized_shape (tuple): Shape for normalization layers.

    Inputs:
        - **query_embedding** (Tensor) - Query embedding tensor of shape (num_res, num_res, num_channels).
        - **templates** (Templates object) - Object containing single template data.
        - **padding_mask_2d** (Tensor) - Padding mask tensor.
        - **multichain_mask_2d** (Tensor) - Mask indicating intra-chain residue pairs.
        - **key** (random.KeyArray) - Random key generator.

    Outputs:
        - **output** (Tensor) - Template embedding tensor of shape (num_res, num_res, num_channels).
    """

    def __init__(
            self,
            config,
            global_config,
            normalized_shape,
            dtype=ms.float32
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        num_channels = self.config.num_channels
        self.query_embedding_norm = bm.LayerNorm(
            normalized_shape, dtype=ms.float32)

        # to be determined the shape of input, output and number of layers
        num_layers = 9
        in_shape_list = [39, (), 31, 31, (), (), (), (), 128]
        ndim_list = [3, 2, 3, 3, 2, 2, 2, 2, 3]
        self.template_pair_embedding = ms.nn.CellList(
            [
                bm.CustomDense(
                    in_shape_list[i], num_channels, weight_init="relu", ndim=ndim_list[i], dtype=dtype
                )
                for i in range(num_layers)
            ]
        )
        self.template_stack = ms.nn.CellList(
            [
                modules.PairFormerIteration(
                    self.config.template_stack, self.global_config, normalized_shape[:-1] + (
                        num_channels,), dtype=dtype
                )
                for _ in range(self.config.template_stack.num_layer)
            ]
        )
        self.output_layer_norm = bm.LayerNorm(
            normalized_shape[:-1] + (num_channels,), dtype=ms.float32)

    def construct(self, query_embedding, templates, padding_mask_2d, multichain_mask_2d, key):
        act = self.construct_input(
            query_embedding, templates, multichain_mask_2d)
        if self.config.template_stack.num_layer:
            for i in range(self.config.template_stack.num_layer):
                act = self.template_stack[i](act, padding_mask_2d)
        act = self.output_layer_norm(act)
        return act

    def construct_input(self, query_embedding, templates, multichain_mask_2d):
        """Construct input for template embedding."""
        # Compute distogram feature for the template.
        dtype = multichain_mask_2d.dtype
        aatype = templates.aatype
        dense_atom_mask = templates.atom_mask
        dense_atom_positions = templates.atom_positions
        dense_atom_positions *= dense_atom_mask[..., None]
        pseudo_beta_positions, pseudo_beta_mask = [ms.Tensor(x) for x in scoring.pseudo_beta_fn(
            templates.aatype, dense_atom_positions, dense_atom_mask
        )]
        pseudo_beta_mask_2d = (
            pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
        )
        pseudo_beta_mask_2d *= multichain_mask_2d
        dgram = dgram_from_positions(
            pseudo_beta_positions, self.config.dgram_features
        )
        dgram *= pseudo_beta_mask_2d[..., None]
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.astype(dtype)
        to_concat = [(dgram, 1), (pseudo_beta_mask_2d, 0)]
        aatype = mint.nn.functional.one_hot(
            aatype.astype(ms.int64),
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
        ).astype(dtype)
        to_concat.append((aatype[None, :, :], 1))
        to_concat.append((aatype[:, None, :], 1))
        template_group_indices = mint.index_select(
            ms.Tensor(protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX),
            0,
            templates.aatype,
        )
        rigid, backbone_mask = make_backbone_rigid(
            geometry.Vec3Array.from_array(dense_atom_positions),
            dense_atom_mask,
            template_group_indices,
        )
        points = rigid.translation
        x = rigid.translation.x.unsqueeze(-1)
        y = rigid.translation.y.unsqueeze(-1)
        z = rigid.translation.z.unsqueeze(-1)
        xx = rigid.rotation.xx.unsqueeze(-1)
        xy = rigid.rotation.xy.unsqueeze(-1)
        xz = rigid.rotation.xz.unsqueeze(-1)
        yx = rigid.rotation.yx.unsqueeze(-1)
        yy = rigid.rotation.yy.unsqueeze(-1)
        yz = rigid.rotation.yz.unsqueeze(-1)
        zx = rigid.rotation.zx.unsqueeze(-1)
        zy = rigid.rotation.zy.unsqueeze(-1)
        zz = rigid.rotation.zz.unsqueeze(-1)
        rigid = geometry.Rigid3Array(geometry.Rot3Array(
            xx, xy, xz, yx, yy, yz, zx, zy, zz), geometry.Vec3Array(x, y, z))
        rigid_vec = rigid.inverse().apply_to_point(points)

        unit_vector = rigid_vec.normalized()
        unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]
        unit_vector = list(unit_vector)

        backbone_mask_2d = (backbone_mask[:, None] * backbone_mask[None, :]).astype(dtype)
        backbone_mask_2d *= multichain_mask_2d
        unit_vector = [x * backbone_mask_2d for x in unit_vector]

        # Note that the backbone_mask takes into account C, CA and N (unlike
        # pseudo beta mask which just needs CB) so we add both masks as features.
        to_concat.extend([(x, 0) for x in unit_vector])
        to_concat.append((backbone_mask_2d, 0))
        query_embedding = self.query_embedding_norm(query_embedding)
        # Allow the template embedder to see the query embedding.  Note this
        # contains the position relative feature, so this is how the network knows
        # which residues are next to each other.
        to_concat.append((query_embedding, 1))

        act = 0
        for i, (x, _) in enumerate(to_concat):
            act += self.template_pair_embedding[i](x)
        return act
