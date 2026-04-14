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

"""Model-side of the input features processing."""
import math
import numpy as np
import mindspore as ms
from mindspore import ops
from alphafold3.constants import residue_names
from alphafold3.model.components import utils


def _grid_keys(key, shape):
    """Generate a grid of rng keys that is consistent with different padding.

    Generate random keys such that the keys will be identical, regardless of
    how much padding is added to any dimension.

    Args:
        key: A PRNG key.
        shape: The shape of the output array of keys that will be generated.

    Returns:
        An array of shape `shape` consisting of random keys.
    """
    if not shape:
        return key

    def partial_bitwise_xor(other):
        return ops.bitwise_xor(key, other)

    def _partial_grid_keys(key):
        return _grid_keys(key, shape[1:])
    new_keys = ms.vmap(partial_bitwise_xor)(
        ops.arange(shape[0])
    )
    return ms.vmap(_partial_grid_keys)(new_keys)


def _padding_consistent_rng(f):
    def inner(key, shape, **kwargs):
        keys = _grid_keys(key, shape)
        out = keys.flatten()
        count = 0
        for key in keys.flatten():
            out[count] = (f((), key))
            count += 1
        return out.reshape(keys.shape)
    return inner


def gumbel_sample(shape):
    uniform_samples = ms.Tensor(np.random.uniform(0.0, 1.0, shape))
    gumbel_samples = -ops.log(-ops.log(uniform_samples))
    return gumbel_samples


def gumbel_argsort_sample_idx(key, logits):
    gumbel = _padding_consistent_rng(gumbel_sample)
    z = gumbel(key, logits.shape)
    perm = ops.argsort(logits + z, axis=-1, descending=False)
    return perm[::-1]


def create_msa_feat(msa):
    msa_1hot = ops.one_hot(msa.rows.astype(
        ms.int64), residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 1)
    deletion_matrix = msa.deletion_matrix
    has_deletion = ops.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (ops.arctan(deletion_matrix / 3.0)
                      * (2.0 / math.pi))[..., None]
    msa_feat = [msa_1hot.astype(deletion_value.dtype), has_deletion, deletion_value]
    return ops.concat(msa_feat, axis=-1)


def truncate_msa_batch(msa, num_msa):
    indices = ops.arange(num_msa)
    return msa.index_msa_rows(indices)


def create_target_feat(batch, append_per_atom_features, dtype=ms.float32):
    token_features = batch.token_features
    target_features = []
    target_features.append(ops.one_hot(
        token_features.aatype.astype(ms.int64),
        residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP).astype(dtype))
    target_features.append(batch.msa.profile)
    target_features.append(batch.msa.deletion_mean[..., None])

    if append_per_atom_features:
        ref_mask = batch.ref_structure.mask
        element_feat = ops.one_hot(batch.ref_structure.element, 128)
        element_feat = utils.mask_mean(
            mask=ref_mask[..., None], value=element_feat, axis=-2, eps=1e-6)
        target_features.append(element_feat)
        pos_feat = batch.ref_structure.positions
        pos_feat = pos_feat.reshape([pos_feat.shape[0], -1])
        target_features.append(pos_feat)
        target_features.append(ref_mask)
    return ops.concat(target_features, axis=-1)


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
    clipped_offset = ops.clip(
        offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    asym_id_same = left_asym_id == right_asym_id
    final_offset = ops.where(
        asym_id_same,
        clipped_offset,
        (2 * max_relative_idx + 1) * ops.ones_like(clipped_offset),
    )
    rel_pos = ops.one_hot(final_offset.astype(
        ms.int64), 2 * max_relative_idx + 2)
    rel_feats.append(rel_pos)

    # Embed relative token index as a one-hot embedding of distance along residue
    token_offset = left_token_index - right_token_index
    clipped_token_offset = ops.clip(
        token_offset + max_relative_idx, min=0, max=2 * max_relative_idx
    )
    residue_same = ops.logical_and((left_asym_id == right_asym_id), (
        left_residue_index == right_residue_index
    ))
    final_token_offset = ops.where(
        residue_same,
        clipped_token_offset,
        (2 * max_relative_idx + 1) * ops.ones_like(clipped_token_offset),
    )
    rel_token = ops.one_hot(final_token_offset.astype(
        ms.int64), 2 * max_relative_idx + 2)
    rel_feats.append(rel_token)

    # Embed same entity ID
    entity_id_same = left_entity_id == right_entity_id
    rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])

    # Embed relative chain ID inside each symmetry class
    rel_sym_id = left_sym_id - right_sym_id

    max_rel_chain = max_relative_chain

    clipped_rel_chain = ops.clip(
        rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
    )

    final_rel_chain = ops.where(
        entity_id_same,
        clipped_rel_chain,
        (2 * max_rel_chain + 1) * ops.ones_like(clipped_rel_chain),
    )
    rel_chain = ops.one_hot(final_rel_chain.astype(
        ms.int64), 2 * max_relative_chain + 2)

    rel_feats.append(rel_chain)

    return ops.concat(rel_feats, axis=-1)


def shuffle_msa(key, msa):
    """Shuffle MSA randomly, return batch with shuffled MSA.

    Args:
    key: rng key for random number generation.
    msa: MSA object to sample msa from.

    Returns:
    Protein with sampled msa.
    """
    key, sample_key = key, key + 1
    # Sample uniformly among sequences with at least one non-masked position.
    logits = (ops.clip(ops.sum(msa.mask, dim=-1), 0.0, 1.0) - 1.0) * 1e6
    index_order = gumbel_argsort_sample_idx(sample_key, logits)
    return msa.index_msa_rows(index_order), sample_key
