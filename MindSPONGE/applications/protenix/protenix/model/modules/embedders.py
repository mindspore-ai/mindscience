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

"""embedders"""

import mindspore as ms

def create_relative_encoding(
    seq_features,
    max_relative_idx,
    max_relative_chain,
):
    """Add relative position encodings."""
    rel_feats = []
    token_index = seq_features.token_index
    residue_index = seq_features.residue_index
    asym_id = seq_features.asym_id
    entity_id = seq_features.entity_id
    sym_id = seq_features.sym_id

    left_asym_id = asym_id[:, None]
    right_asym_id = asym_id[None, :]

    left_residue_index = residue_index[:, None]
    right_residue_index = residue_index[None, :]

    left_token_index = token_index[:, None]
    right_token_index = token_index[None, :]

    left_entity_id = entity_id[:, None]
    right_entity_id = entity_id[None, :]
    left_sym_id = sym_id[:, None]
    right_sym_id = sym_id[None, :]

    # Embed relative positions using a one-hot embedding of distance along chain
    offset = left_residue_index - right_residue_index
    clipped_offset = ms.ops.clip(
        offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    asym_id_same = left_asym_id == right_asym_id
    final_offset = ms.ops.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * ms.ops.ones_like(clipped_offset),
    )
    rel_pos = ms.ops.one_hot(final_offset.astype(
        ms.int64), 2 * max_relative_idx + 2)
    rel_feats.append(rel_pos)

    # Embed relative token index as a one-hot embedding of distance along residue
    token_offset = left_token_index - right_token_index
    clipped_token_offset = ms.ops.clip(
        token_offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    residue_same = ms.ops.logical_and((left_asym_id == right_asym_id), (
        left_residue_index == right_residue_index
    ))
    final_token_offset = ms.ops.where(
        residue_same,
        clipped_token_offset,
        (2 * max_relative_idx + 1) * ms.ops.ones_like(clipped_token_offset),
    )
    rel_token = ms.ops.one_hot(final_token_offset.astype(
        ms.int64), 2 * max_relative_idx + 2)
    rel_feats.append(rel_token)

    # Embed same entity ID
    entity_id_same = left_entity_id == right_entity_id
    rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])

    # Embed relative chain ID inside each symmetry class
    rel_sym_id = left_sym_id - right_sym_id

    max_rel_chain = max_relative_chain

    clipped_rel_chain = ms.ops.clip(
        rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
    )

    final_rel_chain = ms.ops.where(
        entity_id_same,
        clipped_rel_chain,
        (2 * max_rel_chain + 1) * ms.ops.ones_like(clipped_rel_chain),
    )
    rel_chain = ms.ops.one_hot(final_rel_chain.astype(
        ms.int64), 2 * max_relative_chain + 2)

    rel_feats.append(rel_chain)

    return ms.ops.concat(rel_feats, axis=-1)
