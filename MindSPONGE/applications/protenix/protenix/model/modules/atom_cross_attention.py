
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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

"""atom_cross_attention"""

from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, ops, Tensor, mint
from protenix.model import base_config
# from alphafold3.model.components import base_modules as bm
from protenix.model.modules import base_modules as bm
from protenix.model.modules.transformer import CrossAttTransformer
from protenix.model.modules.utils import AggregateAtomToToken, broadcast_token_to_atom
from protenix.model.modules.utils import broadcast_token_to_local_atom_pair, rearrange_qk_to_dense_trunk


@dataclass
class AtomCrossAttEncoderConfig(base_config.BaseConfig):
    per_token_channels: int = 768
    per_atom_channels: int = 128
    atom_transformer: CrossAttTransformer.Config = (
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
    """

    def __init__(self, config):
        super().__init__()
        self.c = config
        self.linear1 = bm.LinearAMP(
            3, self.c.per_atom_channels, has_bias=False)
        self.linear2 = bm.LinearAMP(
            1, self.c.per_atom_channels, has_bias=False)
        self.linear3 = bm.LinearAMP(
            385, self.c.per_atom_channels, has_bias=False)

    def construct(self, batch):
        """_PerAtomConditioning"""
        batch_shape = batch.ref_structure.positions.shape[:-2]
        n_atom = batch.ref_structure.positions.shape[-2]
        # Compute per-atom single conditioning
        # Shape (num_tokens, num_dense, channels)
        act = self.linear1(batch.ref_structure.positions)
        act = act + self.linear2(mint.asinh(batch.ref_structure.charge.reshape(
            *batch_shape, n_atom, 1
        )))
        # Characters are encoded as ASCII code minus 32, so we need 64 classes,
        # to encode all standard ASCII characters between 32 and 96.
        if batch.ref_structure.atom_name_chars.max() != 1:
            atom_name_chars_1hot = ops.one_hot(batch.ref_structure.atom_name_chars, 64,
                                               Tensor(1.0, ms.float32), Tensor(0.0, ms.float32)).astype(act.dtype)
        else:
            atom_name_chars_1hot = batch.ref_structure.atom_name_chars
        # num_token, num_dense, _ = act.shape
        act = act + self.linear3(mint.concat((
            batch.ref_structure.mask.reshape(*batch_shape, n_atom, -1),
            batch.ref_structure.element.reshape(*batch_shape, n_atom, -1),
            atom_name_chars_1hot.reshape(*batch_shape, n_atom, -1),
        ), dim=-1))
        act = act * batch.ref_structure.mask.reshape(*batch_shape, n_atom, 1)
        return act


@dataclass
class AtomCrossAttEncoderOutput:
    """
    AtomCrossAttEncoderOutput
    Args:
        token_act: (num_tokens, ch)
        skip_connection: (num_subsets, num_queries, ch)
        queries_mask: (num_subsets, num_queries)
        queries_single_cond: (num_subsets, num_queries, ch)
        keys_mask: (num_subsets, num_keys)
        keys_single_cond: (num_subsets, num_keys, ch)
        pair_cond: (num_subsets, num_queries, num_keys, ch)
    """
    def __init__(
        self,
        token_act,            # (num_tokens, ch)
        skip_connection,      # (num_subsets, num_queries, ch)
        queries_mask,         # (num_subsets, num_queries)
        queries_single_cond,  # (num_subsets, num_queries, ch)
        keys_mask,            # (num_subsets, num_keys)
        keys_single_cond,     # (num_subsets, num_keys, ch)
        pair_cond,            # (num_subsets, num_queries, num_keys, ch)
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

    def __init__(self, config, global_config, cond_channels=384,
                 n_queries=32, n_keys=128, with_cond=True, dtype=ms.float32):
        super().__init__()
        self.c = config
        self.with_cond = with_cond
        self.dtype = dtype
        self.n_queries = n_queries
        self.n_keys = n_keys
        self._per_atom_conditioning = _PerAtomConditioning(config)
        if self.with_cond:
            self._embed_trunk_single_cond = bm.LinearAMP(
                cond_channels, self.c.per_atom_channels,
                weight_init=global_config.final_init, has_bias=False, dtype=dtype)
            self._lnorm_trunk_single_cond = bm.LayerNorm(
                (cond_channels,), create_beta=False, gamma_init="ones", dtype=dtype)

            self._atom_positions_to_features = bm.LinearAMP(
                3, self.c.per_atom_channels, has_bias=False, dtype=dtype)

            self._embed_trunk_pair_cond = bm.LinearAMP(
                self.c.per_atom_channels, self.c.per_atom_pair_channels,
                weight_init=global_config.final_init, has_bias=False, dtype=dtype)
            self._lnorm_trunk_pair_cond = bm.LayerNorm(
                (self.c.per_atom_channels,), create_beta=False, gamma_init="ones", dtype=dtype)

        self._single_to_pair_cond_row = bm.LinearAMP(
            self.c.per_atom_channels, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        self._single_to_pair_cond_col = bm.LinearAMP(
            self.c.per_atom_channels, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)

        self._embed_pair_offsets = bm.LinearAMP(
            3, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        # self._embed_pair_offsets = bm.CustomDense(3, self.c.per_atom_pair_channels, use_bias=False, ndim=4, dtype=dtype)
        self._embed_pair_distances = bm.LinearAMP(
            1, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        self._embed_pair_offsets_valid = bm.LinearAMP(
            1, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)

        self._pair_mlp_1 = bm.LinearAMP(
            self.c.per_atom_pair_channels, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        self._pair_mlp_2 = bm.LinearAMP(
            self.c.per_atom_pair_channels, self.c.per_atom_pair_channels, has_bias=False, dtype=dtype)
        self._pair_mlp_3 = bm.LinearAMP(self.c.per_atom_pair_channels, self.c.per_atom_pair_channels,
                                        weight_init=global_config.final_init, has_bias=False, dtype=dtype)
        self.relu = nn.ReLU()
        self._project_atom_features_for_aggr = bm.LinearAMP(
            self.c.per_atom_channels, self.c.per_token_channels, has_bias=False, dtype=dtype)

        self._atom_transformer_encoder = CrossAttTransformer(
            self.c.atom_transformer, global_config, in_shape=[
                self.c.per_atom_channels, self.c.per_atom_pair_channels], ndim=2, dtype=dtype
        )
        self.aggr = AggregateAtomToToken(mode='mean')

    def _compute_single_conditioning(self, batch, trunk_single_cond, atom_to_token_idx):
        """Compute single conditioning from atom metadata."""
        token_atoms_single_cond = self._per_atom_conditioning(batch)

        if trunk_single_cond is not None:
            trunk_single_cond = self._embed_trunk_single_cond(
                self._lnorm_trunk_single_cond(trunk_single_cond)
            )
            token_atoms_single_cond = token_atoms_single_cond.unsqueeze(
                dim=-3) + broadcast_token_to_atom(trunk_single_cond, atom_to_token_idx.astype(ms.int32))

        return token_atoms_single_cond

    def _compute_queries_activation(self, token_atoms_act, token_atoms_single_cond):
        """Compute queries activation."""
        if token_atoms_act is None:
            return token_atoms_single_cond

        queries_act = self._atom_positions_to_features(token_atoms_act)
        return queries_act + token_atoms_single_cond

    def _compute_pair_conditioning(self, queries_single_cond, keys_single_cond,
                                   trunk_pair_cond, atom_to_token_idx):
        """Compute pair conditioning from single features."""
        row_act = self._single_to_pair_cond_row(self.relu(queries_single_cond))
        col_act = self._single_to_pair_cond_col(self.relu(keys_single_cond))
        pair_act = row_act[..., None, :] + col_act[..., None, :, :]

        if trunk_pair_cond is not None:
            trunk_pair_cond = self._embed_trunk_pair_cond(
                self._lnorm_trunk_pair_cond(trunk_pair_cond)
            )
            pair_act = pair_act + broadcast_token_to_local_atom_pair(
                z_token=trunk_pair_cond,
                atom_to_token_idx=atom_to_token_idx,
                n_queries=self.n_queries,
                n_keys=self.n_keys,
                compute_mask=False,
            )[0]

        return pair_act

    def _add_geometric_features(self, pair_act, q_trunked_list, k_trunked_list, pad_info):
        """Add geometric features (offsets, distances) to pair activations."""
        offsets_valid = (
            q_trunked_list[1][:, :, None] == k_trunked_list[1][:, None, :]
        ).unsqueeze(-1)
        offsets = q_trunked_list[0][:, :, None, :] - \
            k_trunked_list[0][:, None, :, :]

        pair_act = (self._embed_pair_offsets(offsets) * offsets_valid) * \
            pad_info["mask_trunked"].unsqueeze(dim=-1) + pair_act

        sq_dists = ops.sum(offsets**2, dim=-1)
        pair_act = pair_act + (
            self._embed_pair_distances(
                1.0 / (1 + sq_dists[:, :, :, None])) * offsets_valid
        )

        pair_act = pair_act + \
            self._embed_pair_offsets_valid(offsets_valid.astype(ms.float32))
        return pair_act

    def _apply_pair_mlp(self, pair_act):
        """Apply MLP to pair activations."""
        pair_act2 = self._pair_mlp_1(self.relu(pair_act))
        pair_act2 = self._pair_mlp_2(self.relu(pair_act2))
        return pair_act + self._pair_mlp_3(self.relu(pair_act2))

    def _aggregate_to_tokens(self, queries_act, atom_to_token_idx, batch):
        """Aggregate atom features to tokens."""
        queries_act = mint.nn.functional.relu(
            self._project_atom_features_for_aggr(queries_act))

        if queries_act.ndim == 2:
            return self.aggr(queries_act, atom_to_token_idx.astype(ms.int32),
                             dim_size=int(atom_to_token_idx.max())+1)

        queries_act_list = []
        for i in range(queries_act.shape[0]):
            queries_act_list.append(self.aggr(
                queries_act[i], atom_to_token_idx.astype(ms.int32),
                dim_size=batch.num_tokens))
        return ms.ops.stack(queries_act_list)

    def construct(
        self,
        token_atoms_act,   # (num_tokens, max_atoms_per_token, 3)
        trunk_single_cond,  # (num_tokens, ch)
        trunk_pair_cond,   # (num_tokens, num_tokens, ch)
        batch,  # : feat_batch.Batch,
    ):
        """AtomCrossAttEncoder"""
        atom_to_token_idx = batch.atom_cross_att.token_atoms_to_queries

        # Compute single conditioning
        token_atoms_single_cond = self._compute_single_conditioning(
            batch, trunk_single_cond, atom_to_token_idx)

        # Rearrange to dense trunk layout
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[batch.ref_structure.positions, batch.ref_structure.ref_space_uid],
            k=[batch.ref_structure.positions, batch.ref_structure.ref_space_uid],
            dim_q=[-2, -1], dim_k=[-2, -1],
            n_queries=self.n_queries, n_keys=self.n_keys, compute_mask=True,
        )

        # Compute queries activation
        queries_act = self._compute_queries_activation(
            token_atoms_act, token_atoms_single_cond)

        # Rearrange single conditioning for queries and keys
        queries_single_cond, keys_single_cond, _ = rearrange_qk_to_dense_trunk(
            q=token_atoms_single_cond, k=token_atoms_single_cond,
            dim_q=-2, dim_k=-2,
            n_queries=self.n_queries, n_keys=self.n_keys, compute_mask=False,
        )

        # Compute pair conditioning
        pair_act = self._compute_pair_conditioning(
            queries_single_cond, keys_single_cond, trunk_pair_cond, atom_to_token_idx)

        # Add geometric features
        pair_act = self._add_geometric_features(
            pair_act, q_trunked_list, k_trunked_list, pad_info)

        # Apply MLP
        pair_act = self._apply_pair_mlp(pair_act)

        # Run transformer
        queries_act = self._atom_transformer_encoder(
            queries_act=queries_act,
            queries_single_cond=token_atoms_single_cond,
            pair_cond=pair_act,
        )
        skip_connection = queries_act

        # Aggregate to tokens
        queries_act = self._aggregate_to_tokens(
            queries_act, atom_to_token_idx, batch)

        return AtomCrossAttEncoderOutput(
            token_act=queries_act,
            skip_connection=skip_connection,
            queries_mask=None,
            queries_single_cond=token_atoms_single_cond,
            keys_mask=None,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
        )
