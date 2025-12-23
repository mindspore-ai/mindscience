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

"""Confidence Head."""
from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, ops, Tensor, mint
from protenix.model import base_config
from protenix.model.modules.pairformer import PairFormerIteration
from protenix.model.modules import base_modules as bm
from protenix.model.modules.utils import broadcast_token_to_atom


def _safe_norm(x, keepdims, axis, eps=1e-8):
    return ops.sqrt(eps + ops.sum(ops.square(x), dim=axis, keepdims=keepdims))


@dataclass
class DistogramFeaturesConfig(base_config.BaseConfig):
    # The left edge of the first bin.
    min_bin: float = 3.25
    # The left edge of the final bin. The final bin catches everything larger than
    # `max_bin`.
    max_bin: float = 50.75
    # The number of bins in the distogram.
    num_bins: int = 39


def dgram_from_positions(positions, config):
    """Compute distogram from amino acid positions.

    Args:
        positions: (num_res, 3) Position coordinates.
        config: Distogram bin configuration.

    Returns:
        Distogram with the specified number of bins.
    """
    lower_breaks = mint.linspace(
        config.min_bin, config.max_bin, config.num_bins)
    upper_breaks = mint.concat(
        [lower_breaks[1:], Tensor([1e8], dtype=ms.float32)], dim=-1)
    dist = mint.sum(mint.square(mint.unsqueeze(positions, -2)
                           - mint.unsqueeze(positions, -3)), dim=-1, keepdim=True).sqrt()
    dgram = (dist > lower_breaks).astype(ms.float32) * \
        (dist < upper_breaks).astype(ms.float32)
    return dgram, dist


class ConfidenceHead(nn.Cell):
    """Head to predict the distance errors in a prediction.

    Args:
        config (ConfidenceHead.Config): Configuration for the ConfidenceHead module.
        global_config (base_config.BaseConfig): Global configuration for the model.
        pair_shape (tuple): Shape of the pair features.
        single_shape (tuple): Shape of the single features.
        max_atoms_per_token (int): 24.
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

        pairformer: PairFormerIteration.Config = base_config.autocreate(
            single_attention=base_config.autocreate(),
            single_transition=base_config.autocreate(),
            num_layer=4,
        )
        max_error_bin: float = 31.0
        num_plddt_bins: int = 50
        num_bins: int = 64
        no_embedding_prob: float = 0.2
        confidence_embedding_drop_rate: float = 0.0
        pae: 'ConfidenceHead.PAEConfig' = base_config.autocreate()
        dgram_features: DistogramFeaturesConfig = (
            base_config.autocreate()
        )

    def __init__(self, config, global_config, act_shape, pair_shape,
                 single_shape, max_atoms_per_token, feat_in_channel, out_channel,
                 dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.config = config
        self.global_config = global_config
        self.left_target_feat_project = bm.LinearAMP(
            feat_in_channel, out_channel, has_bias=False, dtype=dtype)
        self.right_target_feat_project = bm.LinearAMP(
            feat_in_channel, out_channel, has_bias=False, dtype=dtype)
        self.distogram_feat_project = bm.LinearAMP(
            DistogramFeaturesConfig.num_bins, out_channel, has_bias=False, dtype=dtype)
        self.pairformer_block = ms.nn.CellList(
            [
                PairFormerIteration(
                    self.config.pairformer, global_config, act_shape,
                    pair_shape, single_shape, with_single=True, dtype=dtype
                )
                for _ in range(self.config.pairformer.num_layer)
            ]
        )
        for layer in self.pairformer_block:
            layer.recompute()
        self.left_half_distance_logits = bm.LinearAMP(
            pair_shape[-1], self.config.num_bins, has_bias=False, dtype=ms.float32)
        self.logits_ln = bm.LayerNorm(pair_shape, dtype=ms.float32)
        self.pae_logits = bm.LinearAMP(
            pair_shape[-1], self.config.pae.num_bins, has_bias=False, dtype=ms.float32)
        self.pae_logits_ln = bm.LayerNorm(pair_shape, dtype=ms.float32)
        self.plddt_logits = bm.CustomDense(
            single_shape[-1], (max_atoms_per_token, self.config.num_plddt_bins), ndim=2, dtype=ms.float32)
        self.plddt_logits_ln = bm.LayerNorm(single_shape, dtype=ms.float32)
        self.experimentally_resolved_logits = bm.CustomDense(
            single_shape[-1], (max_atoms_per_token, 2), ndim=2, dtype=ms.float32)
        self.experimentally_resolved_ln = bm.LayerNorm(
            single_shape, dtype=ms.float32)

        # add for protenix
        self.distance_linear = bm.LinearAMP(
            1, out_channel, dtype=dtype, has_bias=False)
        self.single_lm = bm.LayerNorm(single_shape, dtype=ms.float32)

    def _embed_features(self, dense_atom_positions, distogram_rep_atom_mask,
                        pair_mask, pair_act, target_feat):
        """_embed_features"""
        out = self.left_target_feat_project(target_feat)
        out2 = self.right_target_feat_project(target_feat)[:, None]
        out = out + out2
        positions = dense_atom_positions[..., distogram_rep_atom_mask, :]

        dgram, distance = dgram_from_positions(
            positions, self.config.dgram_features
        )
        dgram = dgram * pair_mask[...,
                                  None] if pair_mask is not None else dgram
        out = out + self.distance_linear(distance.astype(pair_act.dtype))
        out = out + self.distogram_feat_project(dgram.astype(pair_act.dtype))
        return out

    def construct(self, dense_atom_positions, embeddings_single, embeddings_pair, embeddings_target, seq_mask,
                  distogram_rep_atom_mask, batch, drop_embedding=False):
        """construct"""
        embeddings = {
            'pair': embeddings_pair,
            'single': embeddings_single,
            'target_feat': embeddings_target,
        }
        if drop_embedding:
            pair_act = 0 * embeddings['pair'].astype(self.dtype)
        else:
            pair_act = embeddings['pair'].astype(self.dtype)
        seq_mask_cast = seq_mask.astype(
            self.dtype) if seq_mask is not None else None
        pair_mask = seq_mask_cast[:, None] * seq_mask_cast[None,
                                                           :].astype(self.dtype) if seq_mask is not None else None
        single_act = embeddings['single'].astype(self.dtype)
        target_feat = embeddings['target_feat'].astype(self.dtype)
        single_act = self.single_lm(mint.clamp(single_act, -512, 512))
        num_residues = pair_act.shape[0]
        num_pair_channels = pair_act.shape[2]
        n_samples = dense_atom_positions.shape[0]
        confidence_output = []
        pair_act_init = pair_act
        single_act_init = single_act
        for n in range(n_samples):
            pair_act = pair_act_init
            single_act = single_act_init
            pair_act = pair_act + self._embed_features(
                dense_atom_positions[..., n, :, :],
                distogram_rep_atom_mask,
                pair_mask,
                pair_act,
                target_feat,
            )
            for i in range(self.config.pairformer.num_layer):
                pair_act, single_act = self.pairformer_block[i](
                    pair_act, None, single_act)
            pair_act = pair_act.astype(ms.float32)

            if pair_act.shape != (num_residues, num_residues, num_pair_channels):
                raise ValueError("pair_act.shape must be equal to (num_residues, num_residues, num_pair_channels)")

            distance_logits = self.left_half_distance_logits(
                self.logits_ln(pair_act + ops.swapaxes(pair_act, -2, -3)))

            # Predicted aligned error
            # Shape (num_res, num_res, num_bins)
            pae_logits = self.pae_logits(self.pae_logits_ln(pair_act))
            single_act = single_act.astype('float32')

            # pLDDT
            # Shape (num_res, num_atom, num_bins)
            atom_single = broadcast_token_to_atom(
                x_token=single_act, atom_to_token_idx=batch.atom_to_token_idx
            )
            # plddt_logits = self.plddt_logits(self.plddt_logits_ln(atom_single))
            plddt_logits = mint.einsum(
                "...nc,ncb->...nb",
                self.plddt_logits_ln(atom_single),
                self.plddt_logits.weight.transpose(
                    (1, 0, 2))[batch.atom_to_tokatom_idx],
            )
            experimentally_resolved_logits = mint.einsum(
                "...nc,ncb->...nb",
                self.experimentally_resolved_ln(atom_single),
                self.experimentally_resolved_logits.weight.transpose(
                    (1, 0, 2))[batch.atom_to_tokatom_idx],
            )

            confidence_output.append({
                'predicted_lddt': plddt_logits,
                'predicted_experimentally_resolved': experimentally_resolved_logits,
                'predicted_pae': pae_logits,
                'predicted_pde': distance_logits,
            })
        for key in confidence_output[0].keys():
            confidence_output[0][key] = ops.stack(
                [value[key] for value in confidence_output])
        confidence_output = confidence_output[0]
        return confidence_output
