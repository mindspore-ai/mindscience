# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""multimer data preprocess pipeline"""

import collections
import numpy as np
import pandas as pd
import scipy.linalg

from mindsponge1.common import residue_constants
from mindsponge1.data.data_transform import process_unmerged_features, get_crop_size, correct_msa_restypes, \
    make_seq_mask, make_msa_mask, add_padding

REQUIRED_FEATURES = frozenset({
    'aatype', 'all_atom_mask', 'all_atom_positions', 'all_chains_entity_ids',
    'all_crops_all_chains_mask', 'all_crops_all_chains_positions',
    'all_crops_all_chains_residue_ids', 'assembly_num_chains', 'asym_id',
    'bert_mask', 'cluster_bias_mask', 'deletion_matrix', 'deletion_mean',
    'entity_id', 'entity_mask', 'mem_peak', 'msa', 'msa_mask', 'num_alignments',
    'num_templates', 'queue_size', 'residue_index', 'resolution',
    'seq_length', 'seq_mask', 'sym_id', 'template_aatype',
    'template_all_atom_mask', 'template_all_atom_positions',
    "asym_len", "template_sum_probs", "num_sym", "msa_chains" # dyh
})
MSA_FEATURES = ('msa', 'msa_mask', 'deletion_matrix', 'deletion_matrix_int')
TEMPLATE_FEATURES = ('template_aatype', 'template_all_atom_positions',
                     'template_all_atom_mask')
SEQ_FEATURES = ('residue_index', 'aatype', 'all_atom_positions',
                'all_atom_mask', 'seq_mask', 'between_segment_residues',
                'has_alt_locations', 'has_hetatoms', 'asym_id', 'entity_id',
                'sym_id', 'entity_mask', 'deletion_mean',
                'prediction_atom_mask',
                'literature_positions', 'atom_indices_to_group_indices',
                'rigid_group_default_frame', "num_sym") # dyh
CHAIN_FEATURES = ('num_alignments', 'seq_length')
MAX_TEMPLATES = 4
MSA_CROP_SIZE = 2048


def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.

    Args:
      num: A positive integer.

    Returns:
      A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f'Only positive integers allowed, got {num}.')

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord('A')))
        num = num // 26 - 1
    return ''.join(output)


def add_assembly_features(all_chain_features):
    """Add features to distinguish between chains.

    Args:
      all_chain_features: A dictionary which maps chain_id to a dictionary of
        features for each chain.

    Returns:
      all_chain_features: A dictionary which maps strings of the form
        `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
        chains from a homodimer would have keys A_1 and A_2. Two chains from a
        heterodimer would have keys A_1 and B_1.
    """
    # Group the chains by sequence
    seq_to_entity_id = {}
    grouped_chains = {}
    # for chain_id, chain_features in all_chain_features.items():
    for chain_features in all_chain_features:
        seq = str(chain_features['sequence'])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        entity_id_x = seq_to_entity_id.get(seq)
        if entity_id_x not in grouped_chains:
            grouped_chains[entity_id_x] = []
        grouped_chains.get(entity_id_x).append(chain_features)

    new_all_chain_features = []
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        num_sym = len(group_chain_features) # dyh
        for sym_id, chain_features in enumerate(group_chain_features, start=1):
            # new_all_chain_features[
            #     f'{int_id_to_str_id(entity_id)}_{sym_id}'] = chain_features
            seq_length = chain_features['seq_length']
            chain_features['asym_id'] = chain_id * np.ones(seq_length).astype(np.int32)
            chain_features['sym_id'] = sym_id * np.ones(seq_length).astype(np.int32)
            chain_features['entity_id'] = entity_id * np.ones(seq_length).astype(np.int32)
            chain_features["num_sym"] = num_sym * np.ones(seq_length).astype(np.int32) # dyh
            chain_id += 1
            new_all_chain_features.append(chain_features)

    return new_all_chain_features


def _is_homomer_or_monomer(chains) -> bool:
    """Checks if a list of chains represents a homomer/monomer example."""
    # Note that an entity_id of 0 indicates padding.
    num_unique_chains = len(np.unique(np.concatenate(
        [np.unique(chain['entity_id'][chain['entity_id'] > 0]) for chain in chains])))
    return num_unique_chains == 1 or "msa_all_seq" not in chains[0]


def _make_msa_df(chain_features):
    """Makes dataframe with msa features needed for msa pairing."""
    chain_msa = chain_features['msa_all_seq']
    query_seq = chain_msa[0]
    per_seq_similarity = np.sum(query_seq[None] == chain_msa, axis=-1) / float(len(query_seq))
    per_seq_gap = np.sum(chain_msa == 21, axis=-1) / float(len(query_seq))
    msa_df = pd.DataFrame({
        'msa_species_identifiers': chain_features['msa_species_identifiers_all_seq'],
        'msa_row': np.arange(len(chain_features['msa_species_identifiers_all_seq'])),
        'msa_similarity': per_seq_similarity,
        'gap': per_seq_gap
    })
    return msa_df


def _create_species_dict(msa_df):
    """Creates mapping from species to msa dataframe of that species."""
    species_lookup = {}
    for species, species_df in msa_df.groupby('msa_species_identifiers'):
        species_lookup[species] = species_df
    return species_lookup


def _match_rows_by_sequence_similarity(this_species_msa_dfs):
    """Finds MSA sequence pairings across chains based on sequence similarity.

    Each chain's MSA sequences are first sorted by their sequence similarity to
    their respective target sequence. The sequences are then paired, starting
    from the sequences most similar to their target sequence.

    Args:
      this_species_msa_dfs: a list of dataframes containing MSA features for
        sequences for a specific species.

    Returns:
     A list of lists, each containing M indices corresponding to paired MSA rows,
     where M is the number of chains.
    """
    all_paired_msa_rows = []

    num_seqs = [len(species_df) for species_df in this_species_msa_dfs if species_df is not None]
    take_num_seqs = np.min(num_seqs)

    sort_by_similarity = (lambda x: x.sort_values('msa_similarity', axis=0, ascending=False))

    for species_df in this_species_msa_dfs:
        if species_df is not None:
            species_df_sorted = sort_by_similarity(species_df)
            msa_rows = species_df_sorted.msa_row.iloc[:take_num_seqs].values
        else:
            msa_rows = [-1] * take_num_seqs  # take the last 'padding' row
        all_paired_msa_rows.append(msa_rows)
    all_paired_msa_rows = list(np.array(all_paired_msa_rows).transpose())
    return all_paired_msa_rows


def pair_sequences(examples):
    """Returns indices for paired MSA sequences across chains."""

    num_examples = len(examples)

    all_chain_species_dict = []
    common_species = set()
    for chain_features in examples:
        msa_df = _make_msa_df(chain_features)
        species_dict = _create_species_dict(msa_df)
        all_chain_species_dict.append(species_dict)
        common_species.update(set(species_dict))

    common_species = sorted(common_species)
    common_species.remove(b'')  # Remove target sequence species.

    all_paired_msa_rows = [np.zeros(len(examples), int)]
    all_paired_msa_rows_dict = {k: [] for k in range(num_examples)}
    all_paired_msa_rows_dict[num_examples] = [np.zeros(len(examples), int)]

    for species in common_species:
        if not species:
            continue
        this_species_msa_dfs = []
        species_dfs_present = 0
        for species_dict in all_chain_species_dict:
            if species in species_dict:
                this_species_msa_dfs.append(species_dict[species])
                species_dfs_present += 1
            else:
                this_species_msa_dfs.append(None)

        # Skip species that are present in only one chain.
        if species_dfs_present <= 1:
            continue

        if np.any(
                np.array([len(species_df) for species_df in this_species_msa_dfs if
                          isinstance(species_df, pd.DataFrame)]) > 600):
            continue

        paired_msa_rows = _match_rows_by_sequence_similarity(this_species_msa_dfs)
        all_paired_msa_rows.extend(paired_msa_rows)
        all_paired_msa_rows_dict[species_dfs_present].extend(paired_msa_rows)
    all_paired_msa_rows_dict = {
        num_examples: np.array(paired_msa_rows) for num_examples, paired_msa_rows in all_paired_msa_rows_dict.items()
    }
    return all_paired_msa_rows_dict


def reorder_paired_rows(all_paired_msa_rows_dict):
    """Creates a list of indices of paired MSA rows across chains.

    Args:
      all_paired_msa_rows_dict: a mapping from the number of paired chains to the
        paired indices.

    Returns:
      a list of lists, each containing indices of paired MSA rows across chains.
      The paired-index lists are ordered by:
        1) the number of chains in the paired alignment, i.e, all-chain pairings
           will come first.
        2) e-values
    """
    all_paired_msa_rows = []

    for num_pairings in sorted(all_paired_msa_rows_dict, reverse=True):
        paired_rows = all_paired_msa_rows_dict[num_pairings]
        paired_rows_product = abs(np.array([np.prod(rows) for rows in paired_rows]))
        paired_rows_sort_index = np.argsort(paired_rows_product)
        all_paired_msa_rows.extend(paired_rows[paired_rows_sort_index])

    return np.array(all_paired_msa_rows)


def pad_features(feature, feature_name):
    """Add a 'padding' row at the end of the features list.

    The padding row will be selected as a 'paired' row in the case of partial
    alignment - for the chain that doesn't have paired alignment.

    Args:
      feature: The feature to be padded.
      feature_name: The name of the feature to be padded.

    Returns:
      The feature with an additional padding row.
    """
    assert feature.dtype != np.dtype(np.string_)
    if feature_name in ('msa_all_seq', 'msa_mask_all_seq', 'deletion_matrix_all_seq', 'deletion_matrix_int_all_seq'):
        padding = add_padding(feature_name, feature)
    elif feature_name == 'msa_species_identifiers_all_seq':
        padding = [b'']
    else:
        return feature
    feats_padded = np.concatenate([feature, padding], axis=0)
    return feats_padded


def create_paired_features(chains):
    """Returns the original chains with paired NUM_SEQ features.

    Args:
      chains:  A list of feature dictionaries for each chain.

    Returns:
      A list of feature dictionaries with sequence features including only
      rows to be paired.
    """
    chains = list(chains)
    chain_keys = chains[0].keys()

    if len(chains) < 2:
        return chains
    updated_chains = []
    paired_chains_to_paired_row_indices = pair_sequences(chains)
    paired_rows = reorder_paired_rows(paired_chains_to_paired_row_indices)

    for chain_num, chain in enumerate(chains):
        new_chain = {k: v for k, v in chain.items() if '_all_seq' not in k}
        for feature_name in chain_keys:
            if feature_name.endswith('_all_seq'):
                feats_padded = pad_features(chain[feature_name], feature_name)
                new_chain[feature_name] = feats_padded[paired_rows[:, chain_num]]
        new_chain['num_alignments_all_seq'] = np.asarray(len(paired_rows[:, chain_num]))
        updated_chains.append(new_chain)
    return updated_chains


def deduplicate_unpaired_sequences(np_chains):
    """Removes unpaired sequences which duplicate a paired sequence."""

    feature_names = np_chains[0].keys()
    msa_features = MSA_FEATURES
    cache_msa_features = {}
    for chain in np_chains:
        entity_id = int(chain["entity_id"][0])
        if entity_id not in cache_msa_features:
            # Convert the msa_all_seq numpy array to a tuple for hashing.
            sequence_set = set(tuple(s) for s in chain['msa_all_seq'])
            keep_rows = []
            # Go through unpaired MSA seqs and remove any rows that correspond to the
            # sequences that are already present in the paired MSA.
            for row_num, seq in enumerate(chain['msa']):
                if tuple(seq) not in sequence_set:
                    keep_rows.append(row_num)
            new_msa_features = {}
            for feature_name in feature_names:
                if feature_name in msa_features:
                    if keep_rows:
                        new_msa_features[feature_name] = chain[feature_name][keep_rows]
                    else:
                        new_shape = list(chain[feature_name].shape)
                        new_shape[0] = 0
                        new_msa_features[feature_name] = np.zeros(new_shape, dtype=chain[feature_name].dtype)
            cache_msa_features[entity_id] = new_msa_features
        for feature_name in cache_msa_features[entity_id]:
            chain[feature_name] = cache_msa_features[entity_id][feature_name]
        chain['num_alignments'] = np.array(chain['msa'].shape[0], dtype=np.int32)
    return np_chains


def _crop_single_chain(chain,
                       msa_crop_size,
                       max_templates):
    """Crops msa sequences to `msa_crop_size`."""
    msa_size = chain['num_alignments']
    pair_msa_sequences = "num_alignments_all_seq" in chain.keys()
    if pair_msa_sequences:
        msa_crop_size, msa_crop_size_all_seq = get_crop_size(chain["num_alignments_all_seq"], chain["msa_all_seq"],
                                                            msa_crop_size, msa_size)
    else:
        msa_crop_size = np.minimum(msa_size, msa_crop_size)

    include_templates = "template_aatype" in chain and max_templates
    if include_templates:
        num_templates = chain['template_aatype'].shape[0]
        templates_crop_size = np.minimum(num_templates, max_templates)

    for k in chain:
        k_split = k.split('_all_seq')[0]
        if k_split in TEMPLATE_FEATURES:
            chain[k] = chain[k][:templates_crop_size]
        elif k_split in MSA_FEATURES:
            if '_all_seq' in k and pair_msa_sequences:
                chain[k] = chain[k][:msa_crop_size_all_seq]
            else:
                chain[k] = chain[k][:msa_crop_size]

    chain['num_alignments'] = np.asarray(msa_crop_size, dtype=np.int32)
    if include_templates:
        chain['num_templates'] = np.asarray(templates_crop_size, dtype=np.int32)
    if pair_msa_sequences:
        chain['num_alignments_all_seq'] = np.asarray(msa_crop_size_all_seq, dtype=np.int32)
    return chain


def crop_chains(
        chains_list,
        msa_crop_size,
        max_templates):
    """Crops the MSAs for a set of chains.

    Args:
      chains_list: A list of chains to be cropped.
      msa_crop_size: The total number of sequences to crop from the MSA.
      pair_msa_sequences: Whether we are operating in sequence-pairing mode.
      max_templates: The maximum templates to use per chain.

    Returns:
      The chains cropped.
    """

    # Apply the cropping.
    cropped_chains = []
    for chain in chains_list:
        cropped_chain = _crop_single_chain(
            chain,
            msa_crop_size=msa_crop_size,
            max_templates=max_templates)
        cropped_chains.append(cropped_chain)

    return cropped_chains


def _pad_templates(chains,
                   max_templates):
    """For each chain pad the number of templates to a fixed size.

    Args:
      chains: A list of protein chains.
      max_templates: Each chain will be padded to have this many templates.

    Returns:
      The list of chains, updated to have template features padded to
      max_templates.
    """
    for chain in chains:
        for k, v in chain.items():
            if k in TEMPLATE_FEATURES:
                padding = np.zeros_like(v.shape)
                padding[0] = max_templates - v.shape[0]
                padding = [(0, p) for p in padding]
                chain[k] = np.pad(v, padding, mode='constant')
    return chains


def block_diag(*arrs: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Like scipy.linalg.block_diag but with an optional padding value."""
    ones_arrs = [np.ones_like(x) for x in arrs]
    off_diag_mask = 1.0 - scipy.linalg.block_diag(*ones_arrs)
    diag = scipy.linalg.block_diag(*arrs)
    diag += (off_diag_mask * pad_value).astype(diag.dtype)
    return diag


def _merge_features_from_multiple_chains(chains):
    """Merge features from multiple chains.

    Args:
      chains: A list of feature dictionaries that we want to merge.
      pair_msa_sequences: Whether to concatenate MSA features along the
        num_res dimension (if True), or to block diagonalize them (if False).

    Returns:
      A feature dictionary for the merged example.
    """
    merged_example = {}
    for feature_name in chains[0]:
        feats = [x[feature_name] for x in chains]
        feature_name_split = feature_name.split('_all_seq')[0]
        if feature_name_split in MSA_FEATURES:
            if '_all_seq' in feature_name:
                merged_example[feature_name] = np.concatenate(feats, axis=1)
                if feature_name_split == "msa":
                    merged_example["msa_chains_all_seq"] = np.ones(
                        merged_example[feature_name].shape[0]
                    ).reshape(-1, 1)
            else:
                merged_example[feature_name] = block_diag(
                    *feats, pad_value=residue_constants.MSA_PAD_VALUES[feature_name])
                #### dyh
                if feature_name_split == "msa":
                    msa_chains = []
                    for i, feat in enumerate(feats):
                        cur_shape = feat.shape[0]
                        vals = np.ones(cur_shape) * (i + 2)
                        msa_chains.append(vals)
                    merged_example["msa_chains"] = np.concatenate(msa_chains).reshape(
                        -1, 1
                    )
                ####
        elif feature_name_split in SEQ_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=0)
        elif feature_name_split in TEMPLATE_FEATURES:
            merged_example[feature_name] = np.concatenate(feats, axis=1)
        elif feature_name_split in CHAIN_FEATURES:
            merged_example[feature_name] = np.sum(x for x in feats).astype(np.int32)
        else:
            merged_example[feature_name] = feats[0]
    return merged_example


def _concatenate_paired_and_unpaired_features(example):
    """Merges paired and block-diagonalised features."""
    features = MSA_FEATURES + ("msa_chains",) # dyh
    for feature_name in features:
        if feature_name in example:
            feat = example[feature_name]
            feat_all_seq = example[feature_name + '_all_seq']
            merged_feat = np.concatenate([feat_all_seq, feat], axis=0)
            example[feature_name] = merged_feat
    example['num_alignments'] = np.array(example['msa'].shape[0], dtype=np.int32)
    return example


def _correct_post_merged_feats(
        np_example,
        np_chains_list,
        pair_msa_sequences):
    """Adds features that need to be computed/recomputed post merging."""

    np_example['seq_length'] = np.asarray(np_example['aatype'].shape[0], dtype=np.int32)
    np_example['num_alignments'] = np.asarray(np_example['msa'].shape[0], dtype=np.int32)

    if not pair_msa_sequences:
        # Generate a bias that is 1 for the first row of every block in the
        # block diagonal MSA - i.e. make sure the cluster stack always includes
        # the query sequences for each chain (since the first row is the query
        # sequence).
        cluster_bias_masks = []
        for chain in np_chains_list:
            mask = np.zeros(chain['msa'].shape[0])
            mask[0] = 1
            cluster_bias_masks.append(mask)
        np_example['cluster_bias_mask'] = np.concatenate(cluster_bias_masks)

        # Initialize Bert mask with masked out off diagonals.
        msa_masks = [np.ones(x['msa'].shape, dtype=np.float32) for x in np_chains_list] # int8 dyh
        np_example['bert_mask'] = block_diag(*msa_masks, pad_value=0)
    else:
        np_example['cluster_bias_mask'] = np.zeros(np_example['msa'].shape[0])
        np_example['cluster_bias_mask'][0] = 1

        # Initialize Bert mask with masked out off diagonals.
        msa_masks = [np.ones(x['msa'].shape, dtype=np.float32) for x in np_chains_list] # int8 dyh
        msa_masks_all_seq = [np.ones(x['msa_all_seq'].shape, dtype=np.float32) for x in np_chains_list] # int8 dyh

        msa_mask_block_diag = block_diag(*msa_masks, pad_value=0)
        msa_mask_all_seq = np.concatenate(msa_masks_all_seq, axis=1)
        np_example['bert_mask'] = np.concatenate([msa_mask_all_seq, msa_mask_block_diag], axis=0)
    return np_example


def merge_chain_features(np_chains_list,
                         max_templates):
    """Merges features for multiple chains to single FeatureDict.

    Args:
      np_chains_list: List of FeatureDicts for each chain.
      pair_msa_sequences: Whether to merge paired MSAs.
      max_templates: The maximum number of templates to include.

    Returns:
      Single FeatureDict for entire complex.
    """
    np_chains_list = _pad_templates(np_chains_list, max_templates=max_templates)

    np_example = _merge_features_from_multiple_chains(np_chains_list)

    pair_msa_sequences = "msa_all_seq" in np_example.keys()
    if pair_msa_sequences:
        np_example = _concatenate_paired_and_unpaired_features(np_example)

    np_example = _correct_post_merged_feats(
        np_example=np_example,
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences)

    return np_example


def _filter_features(np_example):
    """Filters features of example to only those requested."""
    return {k: v for (k, v) in np_example.items() if k in REQUIRED_FEATURES}


def process_final(np_example):
    """Final processing steps in data pipeline, after merging and pairing."""
    # np_example["msa"] = correct_msa_restypes(np_example["msa"])
    np_example["seq_mask"] = make_seq_mask(np_example["entity_id"])
    np_example["msa_mask"] = make_msa_mask(np_example["msa"], np_example["entity_id"])
    np_example = _filter_features(np_example)
    return np_example


def pair_and_merge(all_chain_features):
    """Runs processing on features to augment, pair and merge.

    Args:
      all_chain_features: A MutableMap of dictionaries of features for each chain.

    Returns:
      A dictionary of features.
    """

    num_chains = len(all_chain_features)
    for chain_features in all_chain_features:
        # Convert deletion matrices to float.
        if "deletion_matrix_int" in chain_features:
            chain_features["deletion_matrix"] = np.asarray(chain_features.pop("deletion_matrix_int"), dtype=np.float32)

        chain_features["deletion_mean"] = np.mean(chain_features["deletion_matrix"], axis=0)

        # Add assembly_num_chains.
        chain_features["assembly_num_chains"] = np.asarray(num_chains)

    # Add entity_mask.
    for chain_features in all_chain_features:
        chain_features["entity_mask"] = (chain_features["entity_id"] != 0).astype(np.int32)

    np_chains_list = all_chain_features

    np_chains_list = crop_chains(
        np_chains_list,
        msa_crop_size=MSA_CROP_SIZE, #2048
        max_templates=MAX_TEMPLATES) #4
    np_example = merge_chain_features(
        np_chains_list=np_chains_list,
        max_templates=MAX_TEMPLATES)
    np_example = process_final(np_example)
    return np_example


def pad_msa(np_example, min_num_seq):
    """ padding features with 0 if seq number less than min_num_seq.

    Args:
      np_example: A feature dict with msa, deletion_matrix, bert_mask, msa_mask and cluster_bias_mask.
      min_num_seq: minimal sequence number

    Returns:
      np_example: padded with 0 features include msa, deletion_matrix, bert_mask, msa_mask and cluster_bias_mask.

    """

    np_example = dict(np_example)
    num_seq = np_example['msa'].shape[0]
    if num_seq < min_num_seq:
        for feat in ('msa', 'deletion_matrix', 'bert_mask', 'msa_mask', "msa_chains"):
            np_example[feat] = np.pad(np_example[feat], ((0, min_num_seq - num_seq), (0, 0)))
        np_example['cluster_bias_mask'] = np.pad(np_example['cluster_bias_mask'], ((0, min_num_seq - num_seq),))
    return np_example


# These four functions are from Unifold Multimer

def empty_template_feats(n_res):
    return {
        "template_aatype": np.zeros((0, n_res)).astype(np.int64),
        "template_all_atom_positions": np.zeros((0, n_res, 37, 3)).astype(np.float32),
        "template_sum_probs": np.zeros((0, 1)).astype(np.float32),
        "template_all_atom_mask": np.zeros((0, n_res, 37)).astype(np.float32),
    }


def uconvert_monomer_features(monomer_features):
    """Reshapes and modifies monomer features for multimer models."""
    if monomer_features["template_aatype"].shape[0] == 0:
        monomer_features.update(
            empty_template_feats(monomer_features["aatype"].shape[0])
        )
    converted = {}
    unnecessary_leading_dim_feats = {
        "sequence",
        "domain_name",
        "num_alignments",
        "seq_length",
    }
    for feature_name, feature in monomer_features.items():
        if feature_name in unnecessary_leading_dim_feats:
            # asarray ensures it's a np.ndarray.
            feature = np.asarray(feature[0], dtype=feature.dtype)
        elif feature_name == "aatype":
            # The multimer model performs the one-hot operation itself.
            feature = np.argmax(feature, axis=-1).astype(np.int32)
        elif feature_name == "template_aatype":
            if feature.shape[0] > 0:
                feature = correct_template_restypes(feature)
        elif feature_name == "template_all_atom_masks":
            feature_name = "template_all_atom_mask"
        elif feature_name == "msa":
            feature = feature.astype(np.uint8)

        if feature_name.endswith("_mask"):
            feature = feature.astype(np.float32)

        converted[feature_name] = feature

    if "deletion_matrix_int" in monomer_features:
        monomer_features["deletion_matrix"] = monomer_features.pop(
            "deletion_matrix_int"
        ).astype(np.float32)

    converted.pop(
        "template_sum_probs"
    )
    return converted


def post_process(np_example):
    np_example = pad_msa(np_example, 512)
    no_dim_keys = [
        "num_alignments",
        "assembly_num_chains",
        "num_templates",
        "seq_length",
        "resolution",
    ]
    for k in no_dim_keys:
        if k in np_example:
            np_example[k] = np_example[k].reshape(-1)
    return np_example


def merge_msas(msa, del_mat, new_msa, new_del_mat):
    cur_msa_set = set([tuple(m) for m in msa])
    new_rows = []
    for i, s in enumerate(new_msa):
        if tuple(s) not in cur_msa_set:
            new_rows.append(i)
    ret_msa = np.concatenate([msa, new_msa[new_rows]], axis=0)
    ret_del_mat = np.concatenate([del_mat, new_del_mat[new_rows]], axis=0)
    return ret_msa, ret_del_mat