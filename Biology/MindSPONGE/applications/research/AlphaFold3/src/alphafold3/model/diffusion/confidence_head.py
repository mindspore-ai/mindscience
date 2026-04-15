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

"""Confidence Head."""
from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, ops
from alphafold3.model import base_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import base_modules as bm
from alphafold3.model.diffusion import modules
from alphafold3.model.diffusion import template_modules


def _safe_norm(x, keepdims, axis, eps=1e-8):
    return ops.sqrt(eps + ops.sum(ops.square(x), dim=axis, keepdims=keepdims))


class ConfidenceHead(nn.Cell):
    """Head to predict the distance errors in a prediction.

    Args:
        config (ConfidenceHead.Config): Configuration for the ConfidenceHead module.
        global_config (base_config.BaseConfig): Global configuration for the model.
        pair_shape (tuple): Shape of the pair features.
        single_shape (tuple): Shape of the single features.
        atom_shape (tuple): Shape of the atom features.
        feat_in_channel (int): Number of input channels for feature projections.
        out_channel (int): Number of output channels for feature projections.

    Inputs:
        - **dense_atom_positions** (Tensor): [N_res, N_atom, 3] array of atom positions.
        - **embeddings** (dict): Dictionary containing pair, single, and target features.
        - **seq_mask** (Tensor): Sequence mask indicating valid residues.
        - **token_atoms_to_pseudo_beta** (Tensor): Pseudo beta information for atom tokens.
        - **asym_id** (Tensor): Asym ID token features.

    Outputs:
        - **predicted_lddt** (Tensor): Predicted LDDT scores for each residue.
        - **predicted_experimentally_resolved** (Tensor): Predicted experimental resolution scores.
        - **full_pde** (Tensor): Full predicted distance errors.
        - **average_pde** (Tensor): Average predicted distance errors.
        - **pae_outputs** (dict): Additional outputs from PAE (Predicted Alignment Error) calculations.
    """
    @dataclass
    class PAEConfig(base_config.BaseConfig):
        max_error_bin: float = 31.0
        num_bins: int = 64

    @dataclass
    class Config(base_config.BaseConfig):
        """Configuration for ConfidenceHead."""

        pairformer: modules.PairFormerIteration.Config = base_config.autocreate(
            single_attention=base_config.autocreate(),
            single_transition=base_config.autocreate(),
            num_layer=4,
        )
        max_error_bin: float = 31.0
        num_plddt_bins: int = 50
        num_bins: int = 64
        no_embedding_prob: float = 0.2
        pae: 'ConfidenceHead.PAEConfig' = base_config.autocreate()
        dgram_features: template_modules.DistogramFeaturesConfig = (
            base_config.autocreate()
        )

    def __init__(self, config, global_config, pair_shape, single_shape, atom_shape,
                 feat_in_channel, out_channel, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.config = config
        self.global_config = global_config
        self.left_target_feat_project = nn.Dense(
            feat_in_channel, out_channel, has_bias=False, dtype=dtype)
        self.right_target_feat_project = nn.Dense(
            feat_in_channel, out_channel, has_bias=False, dtype=dtype)
        self.distogram_feat_project = nn.Dense(
            template_modules.DistogramFeaturesConfig.num_bins, out_channel, has_bias=False, dtype=dtype)
        self.pairformer_block = ms.nn.CellList(
            [
                modules.PairFormerIteration(
                    self.config.pairformer, global_config, pair_shape, single_shape, with_single=True, dtype=dtype
                )
                for _ in range(self.config.pairformer.num_layer)
            ]
        )
        self.left_half_distance_logits = nn.Dense(
            pair_shape[-1], self.config.num_bins, has_bias=False, dtype=ms.float32)
        self.logits_ln = bm.LayerNorm(pair_shape, dtype=ms.float32)
        self.pae_logits = nn.Dense(
            pair_shape[-1], self.config.pae.num_bins, has_bias=False, dtype=ms.float32)
        self.pae_logits_ln = bm.LayerNorm(pair_shape, dtype=ms.float32)
        self.plddt_logits = bm.CustomDense(
            single_shape[-1], (atom_shape[-2], self.config.num_plddt_bins), ndim=2, dtype=ms.float32)
        self.plddt_logits_ln = bm.LayerNorm(single_shape, dtype=ms.float32)
        self.experimentally_resolved_logits = bm.CustomDense(
            single_shape[-1], (atom_shape[-2], 2), ndim=2, dtype=ms.float32)
        self.experimentally_resolved_ln = bm.LayerNorm(single_shape, dtype=ms.float32)

    def _embed_features(self, dense_atom_positions, token_atoms_to_pseude_beta,
                        pair_mask, target_feat):
        out = self.left_target_feat_project(target_feat)
        out2 = self.right_target_feat_project(target_feat)[:, None]
        out = out + out2
        positions = atom_layout.convert_ms(
            token_atoms_to_pseude_beta,
            dense_atom_positions,
            layout_axes=(-3, -2),
        )
        dgram = template_modules.dgram_from_positions(
            positions, self.config.dgram_features, dtype=ms.float32
        )
        dgram *= pair_mask[..., None]
        out += self.distogram_feat_project(dgram)
        return out

    def construct(self, dense_atom_positions, embeddings, seq_mask,
                  token_atoms_to_pseudo_beta, asym_id):
        seq_mask_cast = seq_mask
        pair_mask = seq_mask_cast[:, None] * seq_mask_cast[None, :]
        pair_act = embeddings['pair']
        single_act = embeddings['single']
        target_feat = embeddings['target_feat']
        pair_act += self._embed_features(
            dense_atom_positions,
            token_atoms_to_pseudo_beta,
            pair_mask,
            target_feat,
        )

        for i in range(self.config.pairformer.num_layer):
            pair_act, single_act = self.pairformer_block[i](
                pair_act, pair_mask, single_act, seq_mask)
        pair_act = pair_act.astype(ms.float32)

        # Produce logits to predict a distogram of pairwise distance errors
        # between the input prediction and the ground truth.
        # Shape (num_res, num_res, num_bins)
        left_distance_logits = self.left_half_distance_logits(
            self.logits_ln(pair_act))
        right_distance_logits = left_distance_logits
        distance_logits = left_distance_logits + ops.swapaxes(  # Symmetrize.
            right_distance_logits, -2, -3
        )
        # Shape (num_bins,)
        distance_breaks = ops.linspace(
            0.0, self.config.max_error_bin, self.config.num_bins - 1
        )

        step = distance_breaks[1] - distance_breaks[0]

        # Add half-step to get the center
        bin_centers = distance_breaks + step / 2
        # Add a catch-all bin at the end.
        bin_centers = ops.concat(
            [bin_centers, bin_centers[-1:] + step], axis=0
        )

        distance_probs = ops.softmax(distance_logits, axis=-1)

        pred_distance_error = (
            ops.sum(distance_probs * bin_centers, dim=-1) * pair_mask
        )
        average_pred_distance_error = ops.sum(
            pred_distance_error, dim=[-2, -1]
        ) / ops.sum(pair_mask, dim=[-2, -1])

        # Predicted aligned error
        pae_outputs = {}
        # Shape (num_res, num_res, num_bins)
        pae_logits = self.pae_logits(self.pae_logits_ln(pair_act))
        # Shape (num_bins,)
        pae_breaks = ops.linspace(
            0.0, self.config.pae.max_error_bin, self.config.pae.num_bins - 1
        )
        step = pae_breaks[1] - pae_breaks[0]
        # Add half-step to get the center
        bin_centers = pae_breaks + step / 2
        # Add a catch-all bin at the end.
        bin_centers = ops.concat(
            [bin_centers, bin_centers[-1:] + step], axis=0
        )
        pae_probs = ops.softmax(pae_logits, axis=-1)

        seq_mask_bool = seq_mask.astype(bool)
        pair_mask_bool = seq_mask_bool[:, None] * seq_mask_bool[None, :]
        pae = ops.sum(pae_probs * bin_centers, dim=-1) * pair_mask_bool
        pae_outputs.update({
            'full_pae': pae,
        })

        # The pTM is computed outside of bfloat16 context.
        tmscore_adjusted_pae_global, tmscore_adjusted_pae_interface = (
            self._get_tmscore_adjusted_pae(
                asym_id=asym_id,
                seq_mask=seq_mask,
                pair_mask=pair_mask_bool,
                bin_centers=bin_centers,
                pae_probs=pae_probs,
            )
        )
        pae_outputs.update({
            'tmscore_adjusted_pae_global': tmscore_adjusted_pae_global,
            'tmscore_adjusted_pae_interface': tmscore_adjusted_pae_interface,
        })

        # pLDDT
        # Shape (num_res, num_atom, num_bins)
        plddt_logits = self.plddt_logits(self.plddt_logits_ln(single_act))

        bin_width = 1.0 / self.config.num_plddt_bins
        bin_centers = ops.arange(0.5 * bin_width, 1.0, bin_width)
        predicted_lddt = ops.sum(
            ops.softmax(plddt_logits, axis=-1) * bin_centers, dim=-1
        )
        predicted_lddt = predicted_lddt * 100.0

        # Experimentally resolved
        # Shape (num_res, num_atom, 2)
        experimentally_resolved_logits = self.experimentally_resolved_logits(
            self.experimentally_resolved_ln(single_act)
        )

        predicted_experimentally_resolved = ops.softmax(
            experimentally_resolved_logits, axis=-1
        )[..., 1]

        return {
            'predicted_lddt': predicted_lddt,
            'predicted_experimentally_resolved': predicted_experimentally_resolved,
            'full_pde': pred_distance_error,
            'average_pde': average_pred_distance_error,
            **pae_outputs,
        }

    def _get_tmscore_adjusted_pae(
            self, asym_id, seq_mask, pair_mask, bin_centers, pae_probs,
    ):
        def get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs):
            # Clip to avoid negative/undefined d0.
            clipped_num_res = ops.maximum(num_interface_tokens, 19)

            # Compute d_0(num_res) as defined by TM-score, eqn. (5) in
            # http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
            # Yang & Skolnick "Scoring function for automated
            # assessment of protein structure template quality" 2004.
            d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

            # Make compatible with [num_tokens, num_tokens, num_bins]
            d0 = d0[:, :, None]
            bin_centers = bin_centers[None, None, :]

            # TM-Score term for every bin.
            tm_per_bin = 1.0 / (1 + ops.square(bin_centers) / ops.square(d0))
            # E_distances tm(distance).
            predicted_tm_term = ops.sum(pae_probs * tm_per_bin, dim=-1)
            return predicted_tm_term

        # Interface version
        x = asym_id[None, :] == asym_id[:, None]
        num_chain_tokens = ops.sum(x * pair_mask, dim=-1)
        num_interface_tokens = num_chain_tokens[None,
                                                :] + num_chain_tokens[:, None]
        # Don't double-count within a single chain
        num_interface_tokens -= x * (num_interface_tokens // 2)
        num_interface_tokens = num_interface_tokens * pair_mask

        num_global_tokens = ops.full(
            size=pair_mask.shape, fill_value=seq_mask.sum()
        ).astype(ms.int32)

        global_apae = get_tmscore_adjusted_pae(
            num_global_tokens, bin_centers, pae_probs
        )
        interface_apae = get_tmscore_adjusted_pae(
            num_interface_tokens, bin_centers, pae_probs
        )
        return global_apae, interface_apae
