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
"""data transform MSA TEMPLATE"""
import numpy as np
from scipy.special import softmax
import mindsponge.common.geometry as geometry
from mindsponge.common.residue_constants import chi_angles_mask, chi_pi_periodic, restype_1to3, chi_angles_atoms, \
    atom_order, residue_atom_renaming_swaps, restype_3to1, MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, restype_order, \
    restypes, restype_name_to_atom14_names, atom_types, residue_atoms, STANDARD_ATOM_MASK, restypes_with_x_and_gap, \
    MSA_PAD_VALUES

MS_MIN32 = -2147483648
MS_MAX32 = 2147483647


def one_hot(depth, indices):
    """one hot compute"""
    res = np.eye(depth)[indices.reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


def correct_msa_restypes(msa, deletion_matrix=None, is_evogen=False):
    """Correct MSA restype to have the same order as residue_constants."""
    new_order_list = MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, dtype=msa.dtype)
    msa = new_order[msa]
    if is_evogen:
        msa_input = np.concatenate((msa, deletion_matrix), axis=-1).astype(np.int32)
        result = msa, msa_input
    else:
        result = msa
    return result


def randomly_replace_msa_with_unknown(msa, aatype, replace_proportion):
    """Replace a proportion of the MSA with 'X'."""
    msa_mask = np.random.uniform(size=msa.shape, low=0, high=1) < replace_proportion
    x_idx = 20
    gap_idx = 21
    msa_mask = np.logical_and(msa_mask, msa != gap_idx)
    msa = np.where(msa_mask, np.ones_like(msa) * x_idx, msa)
    aatype_mask = np.random.uniform(size=aatype.shape, low=0, high=1) < replace_proportion
    aatype = np.where(aatype_mask, np.ones_like(aatype) * x_idx, aatype)
    return msa, aatype


def fix_templates_aatype(template_aatype):
    """Fixes aatype encoding of templates."""
    # Map one-hot to indices.
    template_aatype = np.argmax(template_aatype, axis=-1).astype(np.int32)
    # Map hhsearch-aatype to our aatype.
    new_order_list = MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, np.int32)
    template_aatype = new_order[template_aatype]
    return template_aatype


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """compute pseudo beta features from atom positions"""
    is_gly = np.equal(aatype, restype_order['G'])
    ca_idx = atom_order['CA']
    cb_idx = atom_order['CB']
    pseudo_beta = np.where(
        np.tile(is_gly[..., None].astype("int32"), [1] * len(is_gly.shape) + [3]).astype("bool"),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def make_atom14_masks(aatype):
    """create atom 14 position features from aatype"""
    rt_atom14_to_atom37 = []
    rt_atom37_to_atom14 = []
    rt_atom14_mask = []

    for restype in restypes:
        atom_names = restype_name_to_atom14_names.get(restype_1to3.get(restype))

        rt_atom14_to_atom37.append([(atom_order[name] if name else 0) for name in atom_names])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        rt_atom37_to_atom14.append([(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                                    for name in atom_types])

        rt_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    rt_atom14_to_atom37.append([0] * 14)
    rt_atom37_to_atom14.append([0] * 37)
    rt_atom14_mask.append([0.] * 14)

    rt_atom14_to_atom37 = np.array(rt_atom14_to_atom37, np.int32)
    rt_atom37_to_atom14 = np.array(rt_atom37_to_atom14, np.int32)
    rt_atom14_mask = np.array(rt_atom14_mask, np.float32)

    ri_atom14_to_atom37 = rt_atom14_to_atom37[aatype]
    ri_atom14_mask = rt_atom14_mask[aatype]

    atom14_atom_exists = ri_atom14_mask
    ri_atom14_to_atom37 = ri_atom14_to_atom37

    # create the gather indices for mapping back
    ri_atom37_to_atom14 = rt_atom37_to_atom14[aatype]
    ri_atom37_to_atom14 = ri_atom37_to_atom14

    # create the corresponding mask
    restype_atom37_mask = np.zeros([21, 37], np.float32)
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_1to3.get(restype_letter)
        atom_names = residue_atoms.get(restype_name)
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    atom37_atom_exists = restype_atom37_mask[aatype]
    res = [atom14_atom_exists, ri_atom14_to_atom37, ri_atom37_to_atom14, atom37_atom_exists]
    return res


def block_delete_msa_indices(msa, msa_fraction_per_block, randomize_num_blocks, num_blocks):
    """Sample MSA by deleting contiguous blocks.

    Jumper et al. (2021) Suppl. Alg. 1 "MSABlockDeletion"

    Arguments:
    protein: batch dict containing the msa
    config: ConfigDict with parameters

    Returns:
    updated protein
    """

    num_seq = msa.shape[0]
    block_num_seq = np.floor(num_seq * msa_fraction_per_block).astype(np.int32)

    if randomize_num_blocks:
        nb = int(np.random.uniform(0, num_blocks + 1))
    else:
        nb = num_blocks
    del_block_starts = np.random.uniform(0, num_seq, nb).astype(np.int32)
    del_blocks = del_block_starts[:, None] + np.array([_ for _ in range(block_num_seq)]).astype(np.int32)
    del_blocks = np.clip(del_blocks, 0, num_seq - 1)
    del_indices = np.unique(np.sort(np.reshape(del_blocks, (-1,))))

    # Make sure we keep the original sequence
    keep_indices = np.setdiff1d(np.array([_ for _ in range(1, num_seq)]),
                                del_indices)
    keep_indices = np.concatenate([[0], keep_indices], axis=0)
    keep_indices = [int(x) for x in keep_indices]
    return keep_indices


def sample_msa(msa, max_seq):
    """Sample MSA randomly, remaining sequences are stored as `extra_*`."""
    num_seq = msa.shape[0]

    shuffled = list(range(1, num_seq))
    np.random.shuffle(shuffled)
    shuffled.insert(0, 0)
    index_order = np.array(shuffled, np.int32)
    num_sel = min(max_seq, num_seq)

    sel_seq = index_order[:num_sel]
    not_sel_seq = index_order[num_sel:]
    is_sel = num_seq - num_sel
    return is_sel, not_sel_seq, sel_seq


def gumbel_noise(shape):
    """Generate Gumbel Noise of given Shape."""
    epsilon = 1e-6
    uniform_noise = np.random.uniform(0, 1, shape)
    gumbel = -np.log(-np.log(uniform_noise + epsilon) + epsilon)
    return gumbel


def gumbel_argsort_sample_idx(logits):
    """Samples with replacement from a distribution given by 'logits'."""
    z = gumbel_noise(logits.shape)
    return np.argsort(logits + z, axis=-1)[..., ::-1]


def gumbel_permutation(msa_mask, msa_chains=None):
    """gumbel permutation."""
    has_msa = np.sum(msa_mask, axis=-1) > 0
    # default logits is zero
    logits = np.zeros_like(has_msa, dtype=np.float32)
    logits[~has_msa] = -1e6
    # one sample only
    assert len(logits.shape) == 1
    # skip first row
    logits = logits[1:]
    has_msa = has_msa[1:]
    if logits.shape[0] == 0:
        return np.array([0])
    if msa_chains is not None:
        # skip first row
        msa_chains = msa_chains[1:].reshape(-1)
        msa_chains[~has_msa] = 0
        keys, _ = np.unique(msa_chains, return_counts=True)
        num_has_msa = np.array(has_msa.sum())
        num_pair = np.array((msa_chains == 1).sum())
        num_unpair = num_has_msa - num_pair
        num_chains = np.array((keys > 1).sum())
        logits[has_msa] = 1.0 / (num_has_msa + 1e-6)
        logits[~has_msa] = 0
        for k in keys:
            if k > 1:
                cur_mask = msa_chains == k
                cur_cnt = np.array(cur_mask.sum())
                if cur_cnt > 0:
                    logits[cur_mask] *= num_unpair / (num_chains * cur_cnt)
        logits = np.log(logits + 1e-6)
    shuffled = gumbel_argsort_sample_idx(logits) + 1
    return np.concatenate((np.array([0]), shuffled), axis=0)


def sample_msa_v2(msa, msa_chains, msa_mask, max_seq, biased_msa_by_chain=False):
    """Sample MSA randomly in multimer, remaining sequences are stored as `extra_*`."""
    num_seq = msa.shape[0]
    num_sel = min(max_seq, num_seq)
    msa_chain = (msa_chains if biased_msa_by_chain else None)
    index_order = gumbel_permutation(msa_mask, msa_chain)
    num_sel = min(max_seq, num_seq)
    sel_seq = index_order[:num_sel]
    not_sel_seq = index_order[num_sel:]
    is_sel = num_seq - num_sel
    return is_sel, not_sel_seq, sel_seq


def shape_list(x):
    """get the list of dimensions of an array"""
    x = np.array(x)
    if x.ndim is None:
        return x.shape

    static = x.shape
    ret = []
    for _, dimension in enumerate(static):
        ret.append(dimension)
    return ret


def shaped_categorical(probability):
    """get categorical shape"""
    ds = shape_list(probability)
    num_classes = ds[-1]
    flat_probs = np.reshape(probability, (-1, num_classes))
    numbers = list(range(num_classes))
    res = []
    for flat_prob in flat_probs:
        res.append(np.random.choice(numbers, p=flat_prob))
    return np.reshape(np.array(res, np.int32), ds[:-1])


def make_masked_msa(msa, hhblits_profile, uniform_prob, profile_prob, same_prob, replace_fraction, residue_index=None,
                    msa_mask=None, is_evogen=False):
    """create masked msa for BERT on raw MSA features"""

    random_aatype = np.array([0.05] * 20 + [0., 0.], dtype=np.float32)

    probability = uniform_prob * random_aatype + profile_prob * hhblits_profile + same_prob * one_hot(22, msa)

    pad_shapes = [[0, 0] for _ in range(len(probability.shape))]
    pad_shapes[-1][1] = 1
    mask_prob = 1. - profile_prob - same_prob - uniform_prob

    probability = np.pad(probability, pad_shapes, constant_values=(mask_prob,))

    masked_aatype = np.random.uniform(size=msa.shape, low=0, high=1) < replace_fraction

    bert_msa = shaped_categorical(probability)
    bert_msa = np.where(masked_aatype, bert_msa, msa)

    bert_mask = masked_aatype.astype(np.int32)
    true_msa = msa
    msa = bert_msa
    if is_evogen:
        additional_input = np.concatenate((bert_msa[0][:, None], np.asarray(residue_index)[:, None],
                                           msa_mask[0][:, None],
                                           bert_mask[0][:, None]),
                                          axis=-1).astype(np.int32)
        make_masked_msa_result = bert_mask, true_msa, msa, additional_input

    else:
        make_masked_msa_result = bert_mask, true_msa, msa
    return make_masked_msa_result


def share_mask_by_entity(mask_position, entity_id, sym_id, num_sym):
    "share mask by entity"
    entity_id = entity_id
    sym_id = sym_id
    num_sym = num_sym
    unique_entity_ids = np.unique(entity_id)
    first_sym_mask = sym_id == 1
    for cur_entity_id in unique_entity_ids:
        cur_entity_mask = entity_id == cur_entity_id
        cur_num_sym = int(num_sym[cur_entity_mask][0])
        if cur_num_sym > 1:
            cur_sym_mask = first_sym_mask & cur_entity_mask
            cur_sym_bert_mask = mask_position[:, cur_sym_mask]
            mask_position[:, cur_entity_mask] = cur_sym_bert_mask.repeat(cur_num_sym, 0).reshape(
                cur_sym_bert_mask.shape[0], cur_sym_bert_mask.shape[1] * cur_num_sym)
    return mask_position


def gumbel_max_sample(logits):
    """Samples from a probability distribution given by 'logits'."""
    z = gumbel_noise(logits.shape)
    return np.argmax(logits + z, axis=-1)


def make_masked_msa_v2(msa, hhblits_profile, msa_mask, entity_id, sym_id, num_sym,
                       uniform_prob, profile_prob, same_prob,
                       replace_fraction, share_mask=False, bert_mask=None):
    """create masked msa for BERT on raw MSA features"""

    random_aatype = np.array([0.05] * 20 + [0., 0.], dtype=np.float32)
    probability = uniform_prob * random_aatype + profile_prob * hhblits_profile + same_prob * one_hot(22, msa)

    pad_shapes = [[0, 0] for _ in range(len(probability.shape))]
    pad_shapes[-1][1] = 1
    mask_prob = 1.0 - profile_prob - same_prob - uniform_prob
    assert mask_prob >= 0.0
    probability = np.pad(probability, pad_shapes, constant_values=(mask_prob,))
    sh = msa.shape
    mask_position = np.random.rand(*sh) < replace_fraction
    mask_position &= np.array(msa_mask, dtype=bool)
    if bert_mask is not None:
        mask_position &= np.array(bert_mask, dtype=bool)

    if share_mask:
        mask_position = share_mask_by_entity(mask_position, entity_id, sym_id, num_sym)
    logits = np.log(probability + 1e-6)
    bert_msa = gumbel_max_sample(logits)
    bert_msa = np.where(mask_position, bert_msa, msa).astype(np.float32)
    bert_msa *= msa_mask

    mask_position = np.array(mask_position, dtype=np.float32)
    return mask_position, msa, bert_msa


def nearest_neighbor_clusters(msa_mask, msa, extra_msa_mask, extra_msa, gap_agreement_weight=0.):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

    # Determine how much weight we assign to each agreement.  In theory, we could
    # use a full blosum matrix here, but right now let's just down-weight gap
    # agreement because it could be spurious.
    # Never put weight on agreeing on BERT mask
    weights = np.concatenate([np.ones(21), gap_agreement_weight * np.ones(1), np.zeros(1)], 0)

    # Make agreement score as weighted Hamming distance
    sample_one_hot = msa_mask[:, :, None] * one_hot(23, msa)
    num_seq, num_res, _ = sample_one_hot.shape

    array_extra_msa_mask = extra_msa_mask
    if array_extra_msa_mask.any():
        extra_one_hot = extra_msa_mask[:, :, None] * one_hot(23, extra_msa)
        extra_num_seq, _, _ = extra_one_hot.shape

        agreement = np.matmul(
            np.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
            np.reshape(sample_one_hot * weights, [num_seq, num_res * 23]).T)
        # Assign each sequence in the extra sequences to the closest MSA sample
        extra_cluster_assignment = np.argmax(agreement, axis=1)
    else:
        extra_cluster_assignment = np.array([])
    return extra_cluster_assignment


def nearest_neighbor_clusters_v2(msa, msa_mask, extra_msa, extra_msa_mask,
                                 deletion_matrix, extra_deletion_matrix, gap_agreement_weight=0.0):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

    # Determine how much weight we assign to each agreement.  In theory, we could
    # use a full blosum matrix here, but right now let's just down-weight gap
    # agreement because it could be spurious.
    # Never put weight on agreeing on BERT mask.

    weights = np.concatenate([np.ones(21), gap_agreement_weight * np.ones(1), np.zeros(1)], 0)
    msa_one_hot = one_hot(23, msa.astype(np.int32))
    extra_one_hot = one_hot(23, extra_msa)

    msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
    extra_one_hot_masked = extra_msa_mask[:, :, None] * extra_one_hot

    t1 = weights * msa_one_hot_masked
    t1 = np.resize(t1, (t1.shape[0], t1.shape[1] * t1.shape[2]))
    t2 = np.resize(extra_one_hot_masked, (extra_one_hot.shape[0], extra_one_hot.shape[1] * extra_one_hot.shape[2]))
    agreement = t1 @ t2.T
    cluster_assignment = softmax(1e3 * agreement, axis=0)
    cluster_assignment *= np.einsum("mr, nr->mn", msa_mask, extra_msa_mask)

    cluster_count = np.sum(cluster_assignment, axis=-1)
    cluster_count += 1.0  # We always include the sequence itself.

    msa_sum = np.einsum("nm, mrc->nrc", cluster_assignment, extra_one_hot_masked)
    msa_sum += msa_one_hot_masked

    cluster_profile = msa_sum / cluster_count[:, None, None]

    del_sum = np.einsum(
        "nm, mc->nc", cluster_assignment, extra_msa_mask * extra_deletion_matrix
    )
    del_sum += deletion_matrix  # Original sequence.
    cluster_deletion_mean = del_sum / cluster_count[:, None]

    return cluster_profile, cluster_deletion_mean


def summarize_clusters(msa, msa_mask, extra_cluster_assignment, extra_msa_mask, extra_msa, extra_deletion_matrix,
                       deletion_matrix):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = msa.shape[0]

    def csum(x):
        result = []
        for i in range(num_seq):
            result.append(np.sum(x[np.where(extra_cluster_assignment == i)], axis=0))
        return np.array(result)

    mask = extra_msa_mask
    mask_counts = 1e-6 + msa_mask + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * one_hot(23, extra_msa))
    msa_sum += one_hot(23, msa)  # Original sequence
    cluster_profile = msa_sum / mask_counts[:, :, None]

    del msa_sum

    del_sum = csum(mask * extra_deletion_matrix)
    del_sum += deletion_matrix  # Original sequence
    cluster_deletion_mean = del_sum / mask_counts
    del del_sum

    return cluster_profile, cluster_deletion_mean


def crop_extra_msa(extra_msa, max_extra_msa):
    """MSA features are cropped so only `max_extra_msa` sequences are kept."""
    if extra_msa.any():
        num_seq = extra_msa.shape[0]
        num_sel = np.minimum(max_extra_msa, num_seq)
        shuffled = list(range(num_seq))
        np.random.shuffle(shuffled)
        select_indices = shuffled[:num_sel]
        return select_indices
    return None


def make_msa_feat(between_segment_residues, aatype, msa, deletion_matrix, cluster_deletion_mean, cluster_profile,
                  extra_deletion_matrix):
    """Create and concatenate MSA features."""
    # Whether there is a domain break. Always zero for chains, but keeping
    # for compatibility with domain datasets.
    has_break = np.clip(between_segment_residues.astype(np.float32), np.array(0), np.array(1))
    aatype_1hot = one_hot(21, aatype)

    target_feat = [np.expand_dims(has_break, axis=-1), aatype_1hot]

    msa_1hot = one_hot(23, msa)
    has_deletion = np.clip(deletion_matrix, np.array(0), np.array(1))
    deletion_value = np.arctan(deletion_matrix / 3.) * (2. / np.pi)

    msa_feat = [msa_1hot, np.expand_dims(has_deletion, axis=-1), np.expand_dims(deletion_value, axis=-1)]

    if cluster_profile is not None:
        deletion_mean_value = (np.arctan(cluster_deletion_mean / 3.) * (2. / np.pi))
        msa_feat.extend([cluster_profile, np.expand_dims(deletion_mean_value, axis=-1)])
    extra_has_deletion = None
    extra_deletion_value = None
    if extra_deletion_matrix is not None:
        extra_has_deletion = np.clip(extra_deletion_matrix, np.array(0), np.array(1))
        extra_deletion_value = np.arctan(extra_deletion_matrix / 3.) * (2. / np.pi)

    msa_feat = np.concatenate(msa_feat, axis=-1)
    target_feat = np.concatenate(target_feat, axis=-1)
    res = [extra_has_deletion, extra_deletion_value, msa_feat, target_feat]
    return res


def make_msa_feat_v2(msa, deletion_matrix, cluster_deletion_mean, cluster_profile):
    """Create and concatenate MSA features."""
    msa_1hot = one_hot(23, msa.astype(np.int32))
    has_deletion = np.clip(deletion_matrix, 0.0, 1.0)[..., None]
    deletion_value = (np.arctan(deletion_matrix / 3.0) * (2.0 / np.pi))[..., None]

    deletion_mean_value = (np.arctan(cluster_deletion_mean / 3.0) * (2.0 / np.pi))[..., None]

    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
        cluster_profile,
        deletion_mean_value,
    ]
    msa_feat = np.concatenate(msa_feat, axis=-1)
    return msa_feat


def make_extra_msa_feat(extra_msa, extra_deletion_matrix, extra_msa_mask, num_extra_msa):
    # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
    extra_msa = extra_msa[:num_extra_msa]
    deletion_matrix = extra_deletion_matrix[:num_extra_msa]
    has_deletion = np.clip(deletion_matrix, 0.0, 1.0)
    deletion_value = np.arctan(deletion_matrix / 3.0) * (2.0 / np.pi)
    extra_msa_mask = extra_msa_mask[:num_extra_msa]
    return {"extra_msa": extra_msa,
            "extra_msa_mask": extra_msa_mask,
            "extra_msa_has_deletion": has_deletion,
            "extra_msa_deletion_value": deletion_value}


def make_random_seed(size, seed_maker_t, low=MS_MIN32, high=MS_MAX32, random_recycle=False):
    if random_recycle:
        r = np.random.RandomState(seed_maker_t)
        return r.uniform(size=size, low=low, high=high)
    np.random.seed(seed_maker_t)
    return np.random.uniform(size=size, low=low, high=high)


def random_crop_to_size(seq_length, template_mask, crop_size, max_templates,
                        subsample_templates=False, seed=0, random_recycle=False):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    seq_length = seq_length
    seq_length_int = int(seq_length)
    if template_mask is not None:
        num_templates = np.array(template_mask.shape[0], np.int32)
    else:
        num_templates = np.array(0, np.int32)
    num_res_crop_size = np.minimum(seq_length, crop_size)
    num_res_crop_size_int = int(num_res_crop_size)

    # Ensures that the cropping of residues and templates happens in the same way
    # across ensembling iterations.
    # Do not use for randomness that should vary in ensembling.

    if subsample_templates:
        templates_crop_start = int(make_random_seed(size=(), seed_maker_t=seed, low=0, high=num_templates + 1,
                                                    random_recycle=random_recycle))
    else:
        templates_crop_start = 0

    num_templates_crop_size = np.minimum(num_templates - templates_crop_start, max_templates)
    num_templates_crop_size_int = int(num_templates_crop_size)

    num_res_crop_start = int(make_random_seed(size=(), seed_maker_t=seed, low=0,
                                              high=seq_length_int - num_res_crop_size_int + 1,
                                              random_recycle=random_recycle))

    templates_select_indices = np.argsort(make_random_seed(size=[num_templates], seed_maker_t=seed,
                                                           random_recycle=random_recycle))
    res = [num_res_crop_size, num_templates_crop_size_int, num_res_crop_start, num_res_crop_size_int, \
           templates_crop_start, templates_select_indices]
    return res


def atom37_to_torsion_angles(
        aatype: np.ndarray,
        all_atom_pos: np.ndarray,
        all_atom_mask: np.ndarray,
        alt_torsions=False,
        is_multimer=False,
):
    r"""
    This function calculates the seven torsion angles of each residue and encodes them in sine and cosine.
    The order of the seven torsion angles is [pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4]
    Here, pre_omega represents the twist angle between a given amino acid and the previous amino acid.
    The phi represents twist angle between `C-CA-N-(C+1)`, psi represents twist angle between `(N-1)-C-CA-N`.

    Args:
        aatype (numpy.array):           Amino acid type with shape :math:`(batch\_size, N_{res})`.
        all_atom_pos (numpy.array):     Atom37 representation of all atomic coordinates with
                                        shape :math:`(batch\_size, N_{res}, 37, 3)`.
        all_atom_mask (numpy.array):    Atom37 representation of the mask on all atomic coordinates with
                                        shape :math:`(batch\_size, N_{res})`.
        alt_torsions (bool):            Indicates whether to set the sign angle of shielding torsion to zero.
                                        Default: False.
        is_multimer (bool):             It will be True when multimer is used. Default: False

    Returns:
        Dict containing

        - torsion_angles_sin_cos (numpy.array), with shape :math:`(N_{res}, 7, 2)` where
          the final 2 dimensions denote sin and cos respectively. If is_multimer is True, the shape will
          be :math:`(N_{seq}, N_{res}, 7, 2)` .
        - alt_torsion_angles_sin_cos (numpy.array), same as 'torsion_angles_sin_cos', but with the angle shifted
          by pi for all chi angles affected by the naming ambiguities. shape is :math:`(N_{res}, 7, 2)`.
          If is_multimer is True, the shape will be :math:`(N_{seq}, N_{res}, 7, 2)` .
        - torsion_angles_mask (numpy.array), Mask for which chi angles are present. shape is :math:`(N_{res}, 7)` .
          If is_multimer is True, the shape will be :math:`(N_{seq}, N_{res}, 7, 2)` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.data.data_transform import atom37_to_torsion_angles
        >>> n_res = 16
        >>> bs = 1
        >>> aatype = np.random.randn(bs, n_res).astype(np.int32)
        >>> all_atom_pos = np.random.randn(bs, n_res, 37, 3).astype(np.float32)
        >>> all_atom_mask = np.random.randn(bs, n_res, 37).astype(np.float32)
        >>> angle_label_feature = atom37_to_torsion_angles(aatype, all_atom_pos, all_atom_mask)
        >>> print(angle_label_feature.keys())
        dict_keys(['torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask'])
    """

    true_aatype = np.minimum(aatype, 20)

    # get the number residue
    num_batch, num_res = true_aatype.shape

    paddings = np.zeros([num_batch, 1, 37, 3], np.float32)
    padding_atom_pos = np.concatenate([paddings, all_atom_pos[:, :-1, :, :]], axis=1)
    paddings = np.zeros([num_batch, 1, 37], np.float32)
    padding_atom_mask = np.concatenate([paddings, all_atom_mask[:, :-1, :]], axis=1)

    # compute padding atom position for omega, phi and psi
    omega_atom_pos_padding = np.concatenate(
        [padding_atom_pos[..., 1:3, :],
         all_atom_pos[..., 0:2, :]
         ], axis=-2)
    phi_atom_pos_padding = np.concatenate(
        [padding_atom_pos[..., 2:3, :],
         all_atom_pos[..., 0:3, :]
         ], axis=-2)
    psi_atom_pos_padding = np.concatenate(
        [all_atom_pos[..., 0:3, :],
         all_atom_pos[..., 4:5, :]
         ], axis=-2)

    # compute padding atom position mask for omega, phi and psi
    omega_mask_padding = (np.prod(padding_atom_mask[..., 1:3], axis=-1) *
                          np.prod(all_atom_mask[..., 0:2], axis=-1))
    phi_mask_padding = (padding_atom_mask[..., 2] * np.prod(all_atom_mask[..., 0:3], axis=-1))
    psi_mask_padding = (np.prod(all_atom_mask[..., 0:3], axis=-1) * all_atom_mask[..., 4])

    chi_atom_pos_indices = get_chi_atom_pos_indices()
    if is_multimer:
        atom_pos_indices = chi_atom_pos_indices[..., true_aatype, :, :]
    else:
        atom_pos_indices = np_gather_ops(chi_atom_pos_indices, true_aatype, 0, 0)

    chi_atom_pos = np_gather_ops(all_atom_pos, atom_pos_indices, -2, 2, is_multimer)

    angles_mask = list(chi_angles_mask)
    angles_mask.append([0.0, 0.0, 0.0, 0.0])
    angles_mask = np.array(angles_mask)

    if is_multimer:
        chis_mask = angles_mask[true_aatype, :]
    else:
        chis_mask = np_gather_ops(angles_mask, true_aatype, 0, 0)

    chi_angle_atoms_mask = np_gather_ops(all_atom_mask, atom_pos_indices, -1, 2, is_multimer)

    chi_angle_atoms_mask = np.prod(chi_angle_atoms_mask, axis=-1)
    chis_mask = chis_mask * chi_angle_atoms_mask.astype(np.float32)
    torsions_atom_pos_padding = np.concatenate(
        [omega_atom_pos_padding[:, :, None, :, :],
         phi_atom_pos_padding[:, :, None, :, :],
         psi_atom_pos_padding[:, :, None, :, :],
         chi_atom_pos
         ], axis=2)
    torsion_angles_mask_padding = np.concatenate(
        [omega_mask_padding[:, :, None],
         phi_mask_padding[:, :, None],
         psi_mask_padding[:, :, None],
         chis_mask
         ], axis=2)
    torsion_frames = geometry.rigids_from_3_points(
        point_on_neg_x_axis=geometry.vecs_from_tensor(torsions_atom_pos_padding[:, :, :, 1, :]),
        origin=geometry.vecs_from_tensor(torsions_atom_pos_padding[:, :, :, 2, :]),
        point_on_xy_plane=geometry.vecs_from_tensor(torsions_atom_pos_padding[:, :, :, 0, :]))
    inv_torsion_frames = geometry.invert_rigids(torsion_frames)
    vecs = geometry.vecs_from_tensor(torsions_atom_pos_padding[:, :, :, 3, :])
    forth_atom_rel_pos = geometry.rigids_mul_vecs(inv_torsion_frames, vecs)
    torsion_angles_sin_cos = np.stack(
        [forth_atom_rel_pos[2], forth_atom_rel_pos[1]], axis=-1)
    torsion_angles_sin_cos /= np.sqrt(
        np.sum(np.square(torsion_angles_sin_cos), axis=-1, keepdims=True)
        + 1e-8)

    if is_multimer:
        torsion_angles_sin_cos = torsion_angles_sin_cos * np.array(
            [1., 1., -1., 1., 1., 1., 1.])[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
        chi_is_ambiguous = np.array(chi_pi_periodic)[true_aatype, ...]
    else:
        torsion_angles_sin_cos *= np.array(
            [1., 1., -1., 1., 1., 1., 1.])[None, None, :, None]

        chi_is_ambiguous = np_gather_ops(
            np.array(chi_pi_periodic), true_aatype)

    mirror_torsion_angles = np.concatenate(
        [np.ones([num_batch, num_res, 3]),
         1.0 - 2.0 * chi_is_ambiguous], axis=-1)
    alt_torsion_angles_sin_cos = (torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])

    if alt_torsions:
        fix_torsions = np.stack([np.ones(torsion_angles_sin_cos.shape[:-1]),
                                 np.zeros(torsion_angles_sin_cos.shape[:-1])], axis=-1)
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask_padding[
            ..., None] + fix_torsions * (1 - torsion_angles_mask_padding[..., None])
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask_padding[
            ..., None] + fix_torsions * (1 - torsion_angles_mask_padding[..., None])

    if is_multimer:
        return {
            'torsion_angles_sin_cos': torsion_angles_sin_cos,
            'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos,
            'torsion_angles_mask': torsion_angles_mask_padding
        }
    return {
        'torsion_angles_sin_cos': torsion_angles_sin_cos[0],  # (N, 7, 2)
        'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos[0],  # (N, 7, 2)
        'torsion_angles_mask': torsion_angles_mask_padding[0]  # (N, 7)
    }


def atom37_to_frames(
        aatype,
        all_atom_positions,
        all_atom_mask,
        is_affine=False
):
    r"""
    Computes the torsion angle of up to 8 rigid groups for each residue, shape is :math:`[N_{res}, 8, 12]`,
    where 8 is indicates that each residue can be divided into up to 8 rigid groups according to the dependence of
    the atom on the torsion angle, there are 1 backbone frame and 7 side-chain frames.
    For the meaning of 12 ,the first 9 elements are the 9 components of rotation matrix, the last
    3 elements are the 3 component of translation matrix.


    Args:
        aatype(numpy.array):                Amino acid sequence, :math:`[N_{res}]` .
        all_atom_positions(numpy.array):    The coordinates of all atoms, presented as atom37, :math:`[N_{res}, 37, 3]`.
        all_atom_mask(numpy.array):         Mask of all atomic coordinates, :math:`[N_{res}, 37]`.
        is_affine(bool):                    Whether to perform affine, the default value is False.

    Returns:
        Dictionary, the specific content is as follows.

        - **rigidgroups_gt_frames** (numpy.array) - The torsion angle of the 8 rigid body groups for each residue,
          :math:`[N_{res}, 8, 12]`.
        - **rigidgroups_gt_exists** (numpy.array) - The mask of rigidgroups_gt_frames denoting whether the rigid body
          group exists according to the experiment, :math:`[N_{res}, 8]`.
        - **rigidgroups_group_exists** (numpy.array) - Mask denoting whether given group is in principle present
          for given amino acid type, :math:`[N_{res}, 8]` .
        - **rigidgroups_group_is_ambiguous** (numpy.array) - Indicates that the position is chiral symmetry,
          :math:`[N_{res}, 8]` .
        - **rigidgroups_alt_gt_frames** (numpy.array) - 8 Frames with alternative atom renaming
          corresponding to 'all_atom_positions' represented as flat
          12 dimensional array :math:`[N_{res}, 8, 12]` .
        - **backbone_affine_tensor** (numpy.array) - The translation and rotation of the local coordinates of each
          amino acid relative to the global coordinates, :math:`[N_{res}, 7]` , for the last dimension, the first 4
          elements are the affine tensor which contains the rotation information, the last 3 elements are the
          translations in space.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.data import atom37_to_frames
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> aatype = np.ones(193,dtype=np.int32)
        >>> all_atom_positions = np.ones((193,37,3),dtype=np.float32)
        >>> all_atom_mask = np.ones((193,37),dtype=np.int32)
        >>> result = atom37_to_frames(aatype,all_atom_positions,all_atom_mask)
        >>> for key in result.keys():
        >>>     print(key,result[key].shape)
        rigidgroups_gt_frames (193, 8, 12)
        rigidgroups_gt_exists (193, 8)
        rigidgroups_group_exists (193, 8)
        rigidgroups_group_is_ambiguous (193, 8)
        rigidgroups_alt_gt_frames (193, 8, 12)
    """
    aatype_shape = aatype.shape

    flat_aatype = np.reshape(aatype, [-1])
    all_atom_positions = np.reshape(all_atom_positions, [-1, 37, 3])
    all_atom_mask = np.reshape(all_atom_mask, [-1, 37])

    rigid_group_names_res = np.full([21, 8, 3], '', dtype=object)

    # group 0: backbone frame
    rigid_group_names_res[:, 0, :] = ['C', 'CA', 'N']

    # group 3: 'psi'
    rigid_group_names_res[:, 3, :] = ['CA', 'C', 'O']

    # group 4,5,6,7: 'chi1,2,3,4'
    for restype, letter in enumerate(restypes):
        restype_name = restype_1to3[letter]
        for chi_idx in range(4):
            if chi_angles_mask[restype][chi_idx]:
                atom_names = chi_angles_atoms[restype_name][chi_idx]
                rigid_group_names_res[restype, chi_idx + 4, :] = atom_names[1:]

    # create rigid group mask
    rigid_group_mask_res = np.zeros([21, 8], dtype=np.float32)
    rigid_group_mask_res[:, 0] = 1
    rigid_group_mask_res[:, 3] = 1
    rigid_group_mask_res[:20, 4:] = chi_angles_mask

    lookup_table = atom_order.copy()
    lookup_table[''] = 0
    rigid_group_atom37_idx_restype = np.vectorize(lambda x: lookup_table[x])(
        rigid_group_names_res)

    rigid_group_atom37_idx_residx = np_gather_ops(
        rigid_group_atom37_idx_restype, flat_aatype)

    base_atom_pos = np_gather_ops(
        all_atom_positions,
        rigid_group_atom37_idx_residx,
        batch_dims=1)

    gt_frames = geometry.rigids_from_3_points(
        point_on_neg_x_axis=geometry.vecs_from_tensor(base_atom_pos[:, :, 0, :]),
        origin=geometry.vecs_from_tensor(base_atom_pos[:, :, 1, :]),
        point_on_xy_plane=geometry.vecs_from_tensor(base_atom_pos[:, :, 2, :]))

    # get the group mask
    group_masks = np_gather_ops(rigid_group_mask_res, flat_aatype)

    # get the atom mask
    gt_atoms_exists = np_gather_ops(
        all_atom_mask.astype(np.float32),
        rigid_group_atom37_idx_residx,
        batch_dims=1)
    gt_masks = np.min(gt_atoms_exists, axis=-1) * group_masks

    rotations = np.tile(np.eye(3, dtype=np.float32), [8, 1, 1])
    rotations[0, 0, 0] = -1
    rotations[0, 2, 2] = -1
    gt_frames = geometry.rigids_mul_rots(gt_frames, geometry.rots_from_tensor(rotations, use_numpy=True))

    rigid_group_is_ambiguous_res = np.zeros([21, 8], dtype=np.float32)
    rigid_group_rotations_res = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])

    for restype_name, _ in residue_atom_renaming_swaps.items():
        restype = restype_order[restype_3to1[restype_name]]
        chi_idx = int(sum(chi_angles_mask[restype]) - 1)
        rigid_group_is_ambiguous_res[restype, chi_idx + 4] = 1
        rigid_group_rotations_res[restype, chi_idx + 4, 1, 1] = -1
        rigid_group_rotations_res[restype, chi_idx + 4, 2, 2] = -1

    # Gather the ambiguity information for each residue.
    rigid_group_is_ambiguous_res_index = np_gather_ops(
        rigid_group_is_ambiguous_res, flat_aatype)
    rigid_group_ambiguity_rotation_res_index = np_gather_ops(
        rigid_group_rotations_res, flat_aatype)

    # Create the alternative ground truth frames.
    alt_gt_frames = geometry.rigids_mul_rots(
        gt_frames, geometry.rots_from_tensor(rigid_group_ambiguity_rotation_res_index, use_numpy=True))

    gt_frames_flat12 = np.stack(list(gt_frames[0]) + list(gt_frames[1]), axis=-1)
    alt_gt_frames_flat12 = np.stack(list(alt_gt_frames[0]) + list(alt_gt_frames[1]), axis=-1)
    # reshape back to original residue layout
    gt_frames_flat12 = np.reshape(gt_frames_flat12, aatype_shape + (8, 12))
    gt_masks = np.reshape(gt_masks, aatype_shape + (8,))
    group_masks = np.reshape(group_masks, aatype_shape + (8,))
    gt_frames_flat12 = np.reshape(gt_frames_flat12, aatype_shape + (8, 12))
    rigid_group_is_ambiguous_res_index = np.reshape(rigid_group_is_ambiguous_res_index, aatype_shape + (8,))
    alt_gt_frames_flat12 = np.reshape(alt_gt_frames_flat12,
                                      aatype_shape + (8, 12,))
    if not is_affine:
        return {
            'rigidgroups_gt_frames': gt_frames_flat12,  # shape (..., 8, 12)
            'rigidgroups_gt_exists': gt_masks,  # shape (..., 8)
            'rigidgroups_group_exists': group_masks,  # shape (..., 8)
            'rigidgroups_group_is_ambiguous':
                rigid_group_is_ambiguous_res_index,  # shape (..., 8)
            'rigidgroups_alt_gt_frames': alt_gt_frames_flat12,  # shape (..., 8, 12)
        }

    rotation = [[gt_frames[0][0], gt_frames[0][1], gt_frames[0][2]],
                [gt_frames[0][3], gt_frames[0][4], gt_frames[0][5]],
                [gt_frames[0][6], gt_frames[0][7], gt_frames[0][8]]]
    translation = [gt_frames[1][0], gt_frames[1][1], gt_frames[1][2]]
    backbone_affine_tensor = to_tensor(rotation, translation)[:, 0, :]
    return {
        'rigidgroups_gt_frames': gt_frames_flat12,  # shape (..., 8, 12)
        'rigidgroups_gt_exists': gt_masks,  # shape (..., 8)
        'rigidgroups_group_exists': group_masks,  # shape (..., 8)
        'rigidgroups_group_is_ambiguous': rigid_group_is_ambiguous_res_index,  # shape (..., 8)
        'rigidgroups_alt_gt_frames': alt_gt_frames_flat12,  # shape (..., 8, 12)
        'backbone_affine_tensor': backbone_affine_tensor,  # shape (..., 7)
    }


def get_chi_atom_pos_indices():
    """get the atom indices for computing chi angles for all residue types"""
    chi_atom_pos_indices = []
    for residue_name in restypes:
        residue_name = restype_1to3[residue_name]
        residue_chi_angles = chi_angles_atoms[residue_name]
        atom_pos_indices = []
        for chi_angle in residue_chi_angles:
            atom_pos_indices.append([atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_pos_indices)):
            atom_pos_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_pos_indices.append(atom_pos_indices)

    chi_atom_pos_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.array(chi_atom_pos_indices)


def gather(params, indices, axis=0):
    """gather operation"""
    func = lambda p, i: np.take(p, i, axis=axis)
    return func(params, indices)


def np_gather_ops(params, indices, axis=0, batch_dims=0, is_multimer=False):
    """np gather operation"""
    if is_multimer:
        assert axis < 0 or axis - batch_dims >= 0
        ranges = []
        for i, s in enumerate(params.shape[:batch_dims]):
            r = np.arange(s)
            r = np.resize(r, (1,) * i + r.shape + (1,) * (len(indices.shape) - i - 1))
            ranges.append(r)
        remaining_dims = [slice(None) for _ in range(len(params.shape) - batch_dims)]
        remaining_dims[axis - batch_dims if axis >= 0 else axis] = indices
        ranges.extend(remaining_dims)
        return params[tuple(ranges)]

    if batch_dims == 0:
        return gather(params, indices)
    result = []
    if batch_dims == 1:
        for p, i in zip(params, indices):
            axis = axis - batch_dims if axis - batch_dims > 0 else 0
            r = gather(p, i, axis=axis)
            result.append(r)
        return np.stack(result)
    for p, i in zip(params[0], indices[0]):
        r = gather(p, i, axis=axis)
        result.append(r)
    res = np.stack(result)
    return res.reshape((1,) + res.shape)


def rot_to_quat(rot, unstack_inputs=False):
    """transfer the rotation matrix to quaternion matrix"""
    if unstack_inputs:
        rot = [np.moveaxis(x, -1, 0) for x in np.moveaxis(rot, -2, 0)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [[xx + yy + zz, zy - yz, xz - zx, yx - xy],
         [zy - yz, xx - yy - zz, xy + yx, xz + zx],
         [xz - zx, xy + yx, yy - xx - zz, yz + zy],
         [yx - xy, xz + zx, yz + zy, zz - xx - yy]]

    k = (1. / 3.) * np.stack([np.stack(x, axis=-1) for x in k],
                             axis=-2)

    # compute eigenvalues
    _, qs = np.linalg.eigh(k)
    return qs[..., -1]


def to_tensor(rotation, translation):
    """get affine based on rotation and translation"""
    quaternion = rot_to_quat(rotation)
    return np.concatenate(
        [quaternion] +
        [np.expand_dims(x, axis=-1) for x in translation],
        axis=-1)


def convert_monomer_features(chain_id, aatype, template_aatype):
    """Reshapes and modifies monomer features for multimer models."""

    auth_chain_id = np.asarray(chain_id, dtype=np.object_)
    new_order_list = MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    monomer_aatype = np.argmax(aatype, axis=-1).astype(np.int32)
    monomer_template_aatype = np.argmax(template_aatype, axis=-1).astype(np.int32)
    monomer_template_aatype = np.take(new_order_list, monomer_template_aatype.astype(np.int32), axis=0)

    return auth_chain_id, monomer_aatype, monomer_template_aatype


def convert_unnecessary_leading_dim_feats(sequence, domain_name, num_alignments, seq_length):
    """get first dimension data of unnecessary features."""

    monomer_sequence = np.asarray(sequence[0], dtype=sequence.dtype)
    monomer_domain_name = np.asarray(domain_name[0], dtype=domain_name.dtype)
    monomer_num_alignments = np.asarray(num_alignments[0], dtype=num_alignments.dtype)
    monomer_seq_length = np.asarray(seq_length[0], dtype=seq_length.dtype)

    converted_feature = (monomer_sequence, monomer_domain_name, monomer_num_alignments, monomer_seq_length)
    return converted_feature


def process_unmerged_features(deletion_matrix_int, deletion_matrix_int_all_seq, aatype, entity_id, num_chains):
    """Postprocessing stage for per-chain features before merging."""
    # Convert deletion matrices to float.
    deletion_matrix = np.asarray(deletion_matrix_int, dtype=np.float32)
    deletion_matrix_all_seq = np.asarray(deletion_matrix_int_all_seq, dtype=np.float32)

    all_atom_mask = STANDARD_ATOM_MASK[aatype]
    all_atom_mask = all_atom_mask
    all_atom_positions = np.zeros(list(all_atom_mask.shape) + [3])
    deletion_mean = np.mean(deletion_matrix, axis=0)

    # Add assembly_num_chains.
    assembly_num_chains = np.asarray(num_chains)
    entity_mask = (entity_id != 0).astype(np.int32)
    post_feature = (deletion_matrix, deletion_matrix_all_seq, deletion_mean, all_atom_mask, all_atom_positions,
                    assembly_num_chains, entity_mask)

    return post_feature


def get_crop_size(num_alignments_all_seq, msa_all_seq, msa_crop_size, msa_size):
    """get maximum msa crop size

    Args:
        num_alignments_all_seq: num_alignments for all sequence, which record the total number of msa
        msa_all_seq: un-paired sequences for all msa.
        msa_crop_size: The total number of sequences to crop from the MSA.
        msa_size: number of msa

    Returns:
        msa_crop_size: msa sized to be cropped
        msa_crop_size_all_seq: msa_crop_size for features with "_all_seq"

    """

    msa_size_all_seq = num_alignments_all_seq
    msa_crop_size_all_seq = np.minimum(msa_size_all_seq, msa_crop_size // 2)

    # We reduce the number of un-paired sequences, by the number of times a
    # sequence from this chain's MSA is included in the paired MSA.  This keeps
    # the MSA size for each chain roughly constant.
    msa_all_seq = msa_all_seq[:msa_crop_size_all_seq, :]
    num_non_gapped_pairs = np.sum(np.any(msa_all_seq != restypes_with_x_and_gap.index('-'), axis=1))
    num_non_gapped_pairs = np.minimum(num_non_gapped_pairs, msa_crop_size_all_seq)

    # Restrict the unpaired crop size so that paired+unpaired sequences do not
    # exceed msa_seqs_per_chain for each chain.
    max_msa_crop_size = np.maximum(msa_crop_size - num_non_gapped_pairs, 0)
    msa_crop_size = np.minimum(msa_size, max_msa_crop_size)
    return msa_crop_size, msa_crop_size_all_seq


def make_seq_mask(entity_id):
    """seq mask info, True for entity_id > 0, False for entity_id <= 0."""

    seq_mask = (entity_id > 0).astype(np.float32)
    return seq_mask


def make_msa_mask(msa, entity_id):
    """Mask features are all ones, but will later be zero-padded."""

    msa_mask = np.ones_like(msa, dtype=np.float32)

    seq_mask = (entity_id > 0).astype(np.float32)
    msa_mask *= seq_mask[None]

    return msa_mask


def add_padding(feature_name, feature):
    """get padding data with specified shapes of feature"""

    num_res = feature.shape[1]
    padding = MSA_PAD_VALUES.get(feature_name) * np.ones([1, num_res], feature.dtype)
    return padding


def generate_random_sample(cfg, model_config):
    '''generate_random_sample'''
    np.random.seed(0)
    num_noise = model_config.model.latent.num_noise
    latent_dim = model_config.model.latent.latent_dim

    context_true_prob = np.absolute(model_config.train.context_true_prob)
    keep_prob = np.absolute(model_config.train.keep_prob)

    available_msa = int(model_config.train.available_msa_fraction * model_config.train.max_msa_clusters)
    available_msa = min(available_msa, model_config.train.max_msa_clusters)

    evogen_random_data = np.random.normal(
        size=(num_noise, model_config.train.max_msa_clusters, cfg.eval.crop_size, latent_dim)).astype(np.float32)

    # (Nseq,):
    context_mask = np.zeros((model_config.train.max_msa_clusters,), np.int32)
    z1 = np.random.random(model_config.train.max_msa_clusters)
    context_mask = np.asarray([1 if x < context_true_prob else 0 for x in z1], np.int32)
    context_mask[available_msa:] *= 0

    # (Nseq,):
    target_mask = np.zeros((model_config.train.max_msa_clusters,), np.int32)
    z2 = np.random.random(model_config.train.max_msa_clusters)
    target_mask = np.asarray([1 if x < keep_prob else 0 for x in z2], np.int32)

    context_mask[0] = 1
    target_mask[0] = 1

    evogen_context_mask = np.stack((context_mask, target_mask), -1)
    return evogen_random_data, evogen_context_mask


def to_tensor_4x4(feature):
    rots = feature[..., :9]
    trans = feature[..., 9:]
    arrays = np.zeros(feature.shape[:-1] + (4, 4))
    rots = np.reshape(rots, rots.shape[:-1] + (3, 3))
    arrays[..., :3, :3] = rots
    arrays[..., :3, 3] = trans
    arrays[..., 3, 3] = 1
    return arrays
