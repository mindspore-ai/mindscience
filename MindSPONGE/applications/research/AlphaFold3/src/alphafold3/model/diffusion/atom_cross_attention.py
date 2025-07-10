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

from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, ops, Tensor

from alphafold3.model import base_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import base_modules as bm
from alphafold3.model.components import utils
from alphafold3.model.diffusion import diffusion_transformer

@dataclass
class AtomCrossAttEncoderConfig(base_config.BaseConfig):
    per_token_channels: int = 768
    per_atom_channels: int = 128
    atom_transformer: diffusion_transformer.CrossAttTransformer.Config = (
        base_config.autocreate(num_intermediate_factor=2, num_blocks=3)
    )
    per_atom_pair_channels: int = 16


class _PerAtomConditioning(nn.Cell):
    """
    A class to compute per-atom and pairwise conditioning information for structural data.

    Args:
        config: Configuration object containing model parameters.

    Inputs:
        - **batch** (dict) - A dictionary containing structural information:
            - **ref_structure.positions** (Tensor) - Tensor of atomic positions.
            - **ref_structure.mask** (Tensor) - Tensor of masks indicating valid atoms.
            - **ref_structure.element** (Tensor) - Tensor of atomic elements.
            - **ref_structure.charge** (Tensor) - Tensor of atomic charges.
            - **ref_structure.atom_name_chars** (Tensor) - Tensor of atomic name characters.

    Outputs:
        - **act** (Tensor) - Per-atom conditioning information.
        - **pair_act** (Tensor) - Pairwise conditioning information.
    """

    def __init__(self, config):
        super().__init__()
        self.c = config
        self.linear1 = nn.Dense(3, self.c.per_atom_channels, has_bias=False)
        self.linear2 = nn.Dense(1, self.c.per_atom_channels, has_bias=False)
        self.linear3 = nn.Dense(128, self.c.per_atom_channels, has_bias=False)
        self.linear4 = nn.Dense(1, self.c.per_atom_channels, has_bias=False)
        self.linear5 = nn.Dense(256, self.c.per_atom_channels, has_bias=False)
        self.linear_row_act = nn.Dense(
            self.c.per_atom_channels, self.c.per_atom_pair_channels, has_bias=False)
        self.linear_col_act = nn.Dense(
            self.c.per_atom_channels, self.c.per_atom_pair_channels, has_bias=False)
        self.linear_pair_act1 = nn.Dense(
            3, self.c.per_atom_pair_channels, has_bias=False)
        self.linear_pair_act2 = nn.Dense(
            1, self.c.per_atom_pair_channels, has_bias=False)

    @ms.jit
    def construct(self, batch):
        # Compute per-atom single conditioning
        # Shape (num_tokens, num_dense, channels)
        act = self.linear1(batch.ref_structure.positions)
        act += self.linear2(batch.ref_structure.mask[:, :, None])
        # Element is encoded as atomic number if the periodic table, so
        # 128 should be fine.
        act += self.linear3(
            ops.one_hot(batch.ref_structure.element, 128,
                        Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
            .astype(act.dtype))
        act += self.linear4(ops.arcsinh(batch.ref_structure.charge)
                            [:, :, None])
        # Characters are encoded as ASCII code minus 32, so we need 64 classes,
        # to encode all standard ASCII characters between 32 and 96.
        atom_name_chars_1hot = ops.one_hot(batch.ref_structure.atom_name_chars, 64,
                                           Tensor(1.0, ms.float32), Tensor(0.0, ms.float32)).astype(act.dtype)
        num_token, num_dense, _ = act.shape
        act += self.linear5(atom_name_chars_1hot.reshape(num_token, num_dense, -1))
        act *= batch.ref_structure.mask[:, :, None]

        # Compute pair conditioning
        # shape (num_tokens, num_dense, num_dense, channels)
        # Embed single features
        row_act = self.linear_row_act(ops.relu(act))
        col_act = self.linear_col_act(ops.relu(act))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

        # Embed pairwise offsets
        pair_act += self.linear_pair_act1(batch.ref_structure.positions[:, :, None, :]
                                          - batch.ref_structure.positions[:, None, :, :])
        # Embed pairwise inverse squared distances
        sq_dists = ops.sum(ops.square(batch.ref_structure.positions[:, :, None, :]
                                      - batch.ref_structure.positions[:, None, :, :]), dim=-1)
        pair_act += self.linear_pair_act2(1.0 / (1 + sq_dists[:, :, :, None]))
        return act, pair_act

@dataclass
class AtomCrossAttEncoderOutput:
    def __init__(
            self,
            token_act,
            skip_connection,
            queries_mask,
            queries_single_cond,
            keys_mask,
            keys_single_cond,
            pair_cond,
    ):
        self.token_act = token_act
        self.skip_connection = skip_connection
        self.queries_mask = queries_mask
        self.queries_single_cond = queries_single_cond
        self.keys_mask = keys_mask
        self.keys_single_cond = keys_single_cond
        self.pair_cond = pair_cond


class AtomCrossAttEncoder(nn.Cell):
    """Cross-attention on flat atom subsets and mapping to per-token features.

    Args:
        config: Configuration object containing model parameters.
        global_config: Global configuration object with initialization settings.
        name (str): Name of the module.
        cond_channels (int): Number of conditioning channels. Default: ``384``.
        with_cond (bool): Whether to include conditioning layers. Default: ``True``.

    Inputs:
        - **token_atoms_act** (ms.Tensor): Tensor representing token atom activations.
        - **trunk_single_cond** (ms.Tensor): Tensor representing single token conditioning.
        - **trunk_pair_cond** (ms.Tensor): Tensor representing pair token conditioning.
        - **batch** (feat_batch.Batch) : Batch of input data.

    Outputs:
        - **token_act** (ms.Tensor): Activations for tokens after processing.
        - **skip_connection** (ms.Tensor): Skip connection tensor for token queries.
        - **queries_mask** (ms.Tensor): Mask for token queries.
        - **queries_single_cond** (ms.Tensor): Single conditioning for token queries.
        - **keys_mask** (ms.Tensor): Mask for token keys.
        - **keys_single_cond** (ms.Tensor): Single conditioning for token keys.
        - **pair_cond** (ms.Tensor): Pair conditioning tensor.
    """

    def __init__(self, config, global_config, name, cond_channels=384, with_cond=True, dtype=ms.float32):
        super().__init__()
        self.c = config
        self.with_cond = with_cond
        self.dtype = dtype
        self._per_atom_conditioning = _PerAtomConditioning(config)
        if self.with_cond:
            self._embed_trunk_single_cond = nn.Dense(cond_channels, self.c.per_atom_channels,
                                                     weight_init=global_config.final_init, has_bias=False, dtype=dtype)
            self._lnorm_trunk_single_cond = bm.LayerNorm((cond_channels,),
                                                         create_beta=False, gamma_init="ones", dtype=dtype)
            self._atom_positions_to_features = nn.Dense(3, self.c.per_atom_channels, has_bias=False, dtype=dtype)
            self._embed_trunk_pair_cond = nn.Dense(self.c.per_atom_channels, self.c.per_atom_pair_channels,
                                                   weight_init=global_config.final_init, has_bias=False, dtype=dtype)
            self._lnorm_trunk_pair_cond = bm.LayerNorm((self.c.per_atom_channels,), create_beta=False,
                                                       gamma_init="ones", dtype=dtype)

        self._single_to_pair_cond_row = nn.Dense(
            self.c.per_atom_channels, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        self._single_to_pair_cond_col = nn.Dense(
            self.c.per_atom_channels, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)

        self._embed_pair_offsets = nn.Dense(
            3, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        self._embed_pair_distances = nn.Dense(
            1, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        self._embed_pair_offsets_valid = nn.Dense(
            1, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)

        self._pair_mlp_1 = nn.Dense(
            self.c.per_atom_pair_channels, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        self._pair_mlp_2 = nn.Dense(
            self.c.per_atom_pair_channels, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        self._pair_mlp_3 = nn.Dense(self.c.per_atom_pair_channels, self.c.per_atom_pair_channels,
                                    weight_init=global_config.final_init, has_bias=False, dtype=dtype)
        self.relu = nn.ReLU()
        self._project_atom_features_for_aggr = nn.Dense(
            self.c.per_atom_channels, self.c.per_token_channels, has_bias=False, dtype=dtype)

        self._atom_transformer_encoder = diffusion_transformer.CrossAttTransformer(
            self.c.atom_transformer, global_config, in_shape=[
                self.c.per_atom_channels, self.c.per_atom_pair_channels], dtype=dtype
        )

    def construct(
            self,
            token_atoms_act,
            trunk_single_cond,
            trunk_pair_cond,
            batch,
    ):
        # Compute single conditioning from atom meta data and convert to queries
        # layout.
        token_atoms_single_cond, _ = self._per_atom_conditioning(
            batch)
        token_atoms_mask = batch.predicted_structure_info.atom_mask
        queries_single_cond = atom_layout.convert_ms(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_single_cond,
            layout_axes=(-3, -2),
        )
        queries_mask = atom_layout.convert_ms(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_mask,
            layout_axes=(-2, -1),
        )

        # If provided, broadcast single conditioning from trunk to all queries
        if trunk_single_cond is not None:
            trunk_single_cond = self._embed_trunk_single_cond(
                self._lnorm_trunk_single_cond(
                    trunk_single_cond)
            )
            queries_single_cond += atom_layout.convert_ms(
                batch.atom_cross_att.tokens_to_queries,
                trunk_single_cond,
                layout_axes=(-2,),
            )

        if token_atoms_act is None:
            # if no token_atoms_act is given (e.g. begin of evoformer), we use the
            # static conditioning only
            queries_act = queries_single_cond
        else:
            # Convert token_atoms_act to queries layout and map to per_atom_channels
            queries_act = atom_layout.convert_ms(
                batch.atom_cross_att.token_atoms_to_queries,
                token_atoms_act,
                layout_axes=(-3, -2),
            )
            queries_act = self._atom_positions_to_features(
                queries_act)
            queries_act *= queries_mask[..., None]
            queries_act += queries_single_cond

        # Gather the keys from the queries.
        keys_single_cond = atom_layout.convert_ms(
            batch.atom_cross_att.queries_to_keys, queries_single_cond, layout_axes=(
                -3, -2),
        )
        keys_mask = atom_layout.convert_ms(
            batch.atom_cross_att.queries_to_keys, queries_mask, layout_axes=(
                -2, -1)
        )

        # Embed single features into the pair conditioning.
        row_act = self._single_to_pair_cond_row(
            self.relu(queries_single_cond))
        pair_cond_keys_input = atom_layout.convert_ms(
            batch.atom_cross_att.queries_to_keys, queries_single_cond, layout_axes=(
                -3, -2),
        )
        col_act = self._single_to_pair_cond_col(
            self.relu(pair_cond_keys_input))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

        if trunk_pair_cond is not None:
            # If provided, broadcast the pair conditioning for the trunk (evoformer
            # pairs) to the atom pair activations. This should boost ligands, but also
            # help for cross attention within proteins, because we always have atoms
            # from multiple residues in a subset.
            # Map trunk pair conditioning to per_atom_pair_channels
            trunk_pair_cond = self._embed_trunk_pair_cond(
                self._lnorm_trunk_pair_cond(
                    trunk_pair_cond)
            )

            # Create the GatherInfo into a flattened trunk_pair_cond from the
            # queries and keys gather infos.
            num_tokens = trunk_pair_cond.shape[0]
            tokens_to_queries = batch.atom_cross_att.tokens_to_queries
            tokens_to_keys = batch.atom_cross_att.tokens_to_keys

            # Gather the conditioning and add it to the atom-pair activations.
            gather_idxs = Tensor(num_tokens * tokens_to_queries.gather_idxs[:, :, None] +
                                 tokens_to_keys.gather_idxs[:, None, :])
            gather_mask = ops.logical_and(tokens_to_queries.gather_mask[:, :, None],
                                          tokens_to_keys.gather_mask[:, None, :])
            input_shape = Tensor((num_tokens, num_tokens))
            trunk_pair_to_atom_pair = atom_layout.GatherInfo(gather_idxs=gather_idxs,
                                                             gather_mask=gather_mask,
                                                             input_shape=input_shape)
            pair_act += atom_layout.convert_ms(
                trunk_pair_to_atom_pair, trunk_pair_cond, layout_axes=(-3, -2)
            )

        # Embed pairwise offsets
        queries_ref_pos = atom_layout.convert_ms(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.positions,
            layout_axes=(-3, -2),
        )
        queries_ref_space_uid = atom_layout.convert_ms(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.ref_space_uid,
            layout_axes=(-2, -1),
        )
        keys_ref_pos = atom_layout.convert_ms(
            batch.atom_cross_att.queries_to_keys,
            queries_ref_pos,
            layout_axes=(-3, -2),
        )
        keys_ref_space_uid = atom_layout.convert_ms(
            batch.atom_cross_att.queries_to_keys,
            batch.ref_structure.ref_space_uid,
            layout_axes=(-2, -1),
        )

        offsets_valid = (
            queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
        )
        offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
        pair_act += (self._embed_pair_offsets(offsets)
                     * offsets_valid[:, :, :, None])

        # Embed pairwise inverse squared distances
        sq_dists = ops.sum(ops.square(offsets), dim=-1)
        pair_act += (
            self._embed_pair_distances(1.0 / (1 + sq_dists[:, :, :, None]))
            * offsets_valid[:, :, :, None]
        )

        # Embed offsets valid mask
        pair_act += self._embed_pair_offsets_valid(
            offsets_valid[:, :, :, None].astype(ms.float32))

        # Run a small MLP on the pair acitvations
        pair_act2 = self._pair_mlp_1(self.relu(pair_act))
        pair_act2 = self._pair_mlp_2(self.relu(pair_act2))
        pair_act += self._pair_mlp_3(self.relu(pair_act2))

        # Run the atom cross attention transformer.
        queries_act = self._atom_transformer_encoder(
            queries_act=queries_act,
            queries_mask=queries_mask,
            queries_to_keys=batch.atom_cross_att.queries_to_keys,
            keys_mask=keys_mask,
            queries_single_cond=queries_single_cond,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
        )
        queries_act *= queries_mask[..., None]
        skip_connection = queries_act

        # convert back to token-atom layout and aggregate to tokens
        queries_act = self._project_atom_features_for_aggr(queries_act)
        token_atoms_act = atom_layout.convert_ms(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_act,
            layout_axes=(-3, -2),
        )
        token_act = utils.mask_mean(
            token_atoms_mask[..., None], self.relu(token_atoms_act), axis=-2
        )

        return AtomCrossAttEncoderOutput(
            token_act=token_act,
            skip_connection=skip_connection,
            queries_mask=queries_mask,
            queries_single_cond=queries_single_cond,
            keys_mask=keys_mask,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
        )

@dataclass
class AtomCrossAttDecoderConfig(base_config.BaseConfig):
    per_token_channels: int = 768
    per_atom_channels: int = 128
    per_atom_pair_channels: int = 16
    atom_transformer: diffusion_transformer.CrossAttTransformer.Config = (
        base_config.autocreate(num_intermediate_factor=2, num_blocks=3)
    )


class AtomCrossAttDecoder(nn.Cell):
    """Mapping to per-atom features and self-attention on subsets.

    Args:
        config: Configuration object containing model parameters.
        global_config: Global configuration object with additional parameters.
        name (str): Name of the decoder.  Default: ``None``.

    Inputs:
        - **token_act** (Tensor) - Tensor representing token activations.
        - **enc** (AtomCrossAttEncoderOutput) - Output from the encoder containing necessary features and masks.
        - **batch** (feat_batch.Batch) - Batch containing atom cross attention features.

    Outputs:
        - **position_update** (Tensor) - Tensor representing the updated positions after processing.
    """

    def __init__(self, config, global_config, name, dtype=ms.float32):
        super().__init__()
        self.c = config
        self._project_token_features_for_broadcast = nn.Dense(
            self.c.per_token_channels, self.c.per_atom_channels, has_bias=False, dtype=dtype)
        self._atom_features_layer_norm = bm.LayerNorm(
            (self.c.per_atom_channels,), create_beta=False, gamma_init="ones", dtype=dtype)
        self._atom_features_to_position_update = nn.Dense(
            self.c.per_atom_channels, 3, weight_init=global_config.final_init, has_bias=False, dtype=dtype)
        self._atom_transformer_decoder = diffusion_transformer.CrossAttTransformer(
            self.c.atom_transformer, global_config, in_shape=[
                self.c.per_atom_channels, self.c.per_atom_pair_channels], dtype=dtype
        )

    def construct(
            self,
            token_act,
            enc,
            batch,
    ):
        # map per-token act down to per_atom channels
        token_act = self._project_token_features_for_broadcast(token_act)
        # Broadcast to token-atoms layout and convert to queries layout.
        num_token, max_atoms_per_token = (
            batch.atom_cross_att.queries_to_token_atoms.shape
        )
        token_atom_act = ops.broadcast_to(
            token_act[:, None, :],
            (num_token, max_atoms_per_token, self.c.per_atom_channels),
        )
        queries_act = atom_layout.convert_ms(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atom_act,
            layout_axes=(-3, -2),
        )
        queries_act += enc.skip_connection
        queries_act *= enc.queries_mask[..., None]

        # Run the atom cross attention transformer.
        queries_act = self._atom_transformer_decoder(
            queries_act=queries_act,
            queries_mask=enc.queries_mask,
            queries_to_keys=batch.atom_cross_att.queries_to_keys,
            keys_mask=enc.keys_mask,
            queries_single_cond=enc.queries_single_cond,
            keys_single_cond=enc.keys_single_cond,
            pair_cond=enc.pair_cond,
        )

        queries_act *= enc.queries_mask[..., None]
        queries_position_update = self._atom_features_to_position_update(
            self._atom_features_layer_norm(queries_act)
        )
        position_update = atom_layout.convert_ms(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_position_update,
            layout_axes=(-3, -2),
        )
        return position_update
