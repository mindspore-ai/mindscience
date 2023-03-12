# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""megafold_data"""

import numpy as np
from ....common import residue_constants
from ...dataset import curry1
from ....data.data_transform import one_hot
from ....common.utils import make_atom14_positions
from ....data.data_transform import make_atom14_masks
from ....common import geometry
from .megafold_feature import NUM_RES, NUM_MSA_SEQ, NUM_EXTRA_SEQ, NUM_TEMPLATES


@curry1
def dict_filter_key(feature=None, feature_list=None):
    "dict_filter_key"
    feature = {k: v for k, v in feature.items() if k in feature_list}
    return feature


@curry1
def dict_replace_key(feature=None, replaced_key=None):
    "dict_replace_key"
    assert len(replaced_key) == 2
    origin_key, new_key = replaced_key
    if origin_key in feature:
        feature[new_key] = feature.pop(origin_key)
    return feature


@curry1
def dict_cast(feature=None, cast_type=None, filtered_list=None):
    "dict_cast"
    assert len(cast_type) == 2
    origin_type = cast_type[0]
    new_type = cast_type[1]
    for k, v in feature.items():
        if k not in filtered_list:
            if v.dtype == origin_type:
                feature[k] = v.astype(new_type)
    return feature


@curry1
def dict_suqeeze(feature=None, filter_list=None, axis=None):
    "dict_suqeeze"
    for k in filter_list:
        if k in feature:
            n_dim = feature[k].shape[axis]
            if isinstance(n_dim, int) and n_dim == 1:
                feature[k] = np.squeeze(feature[k], axis=axis)
    return feature


@curry1
def dict_take(feature=None, filter_list=None, axis=None):
    "dict_take"
    for k in filter_list:
        if k in feature:
            feature[k] = feature[k][axis]
    return feature


@curry1
def dict_del_key(feature=None, filter_list=None):
    "dict_del_key"
    for k in filter_list:
        if k in feature:
            del feature[k]
    return feature


@curry1
def one_hot_convert(feature=None, key=None, axis=None):
    "one_hot_convert"
    if key in feature and feature[key].shape[0] > 0:
        feature[key] = np.argmax(feature[key], axis=axis)
    return feature


@curry1
def correct_restypes(feature=None, key=None):
    "correct_restypes"
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, dtype=feature[key].dtype)
    feature[key] = new_order[feature[key]]
    return feature


@curry1
def msa_randomly_replace_with_unknown(feature=None, replace_proportion=None, seed=None):
    """Replace a proportion of the MSA with 'X'.
    changed key : msa, aatype
    """
    if seed is not None:
        np.random.seed(seed)
    msa_mask = np.random.uniform(size=feature['msa'].shape, low=0, high=1) < replace_proportion
    x_idx = 20
    gap_idx = 21
    msa_mask = np.logical_and(msa_mask, feature['msa'] != gap_idx)
    feature['msa'] = np.where(msa_mask, np.ones_like(feature['msa']) * x_idx, feature['msa'])
    aatype_mask = np.random.uniform(size=feature['aatype'].shape, low=0, high=1) \
        < replace_proportion
    feature['aatype'] = np.where(aatype_mask,
                                 np.ones_like(feature['aatype']) * x_idx,
                                 feature['aatype'])
    if seed is not None:
        np.random.seed()
    return feature


@curry1
def msa_block_deletion(feature=None, msa_feature_list=None, msa_fraction_per_block=None,
                       randomize_num_blocks=None, num_blocks=None, seed=None):
    "msa_block_deletion"
    if seed is not None:
        np.random.seed(seed)
    num_seq = feature['msa'].shape[0]
    block_num_seq = np.floor(num_seq * msa_fraction_per_block).astype(np.int32)

    if randomize_num_blocks:
        nb = int(np.random.uniform(0, num_blocks + 1))
    else:
        nb = num_blocks
    del_block_starts = np.random.uniform(0, num_seq, nb).astype(np.int32)
    del_blocks = del_block_starts[:, None] + np.array(list(range(block_num_seq))).astype(np.int32)
    del_blocks = np.clip(del_blocks, 0, num_seq - 1)
    del_indices = np.unique(np.sort(np.reshape(del_blocks, (-1,))))

    # Make sure we keep the original sequence
    keep_indices = np.setdiff1d(np.array([_ for _ in range(1, num_seq)]),
                                del_indices)
    keep_indices = np.concatenate([[0], keep_indices], axis=0)
    keep_indices = [int(x) for x in keep_indices]
    feature = dict_take(msa_feature_list, axis=keep_indices)(feature)
    if seed is not None:
        np.random.seed()
    return feature


@curry1
def msa_sample(feature=None, msa_feature_list=None, keep_extra=None, max_msa_clusters=None,
               seed=None):
    "msa_sample"
    if seed is not None:
        np.random.seed(seed)
    num_seq = feature['msa'].shape[0]
    shuffled = list(range(1, num_seq))
    np.random.shuffle(shuffled)
    shuffled.insert(0, 0)
    index_order = np.array(shuffled, np.int32)
    num_sel = min(max_msa_clusters, num_seq)

    sel_seq = index_order[:num_sel]
    not_sel_seq = index_order[num_sel:]
    is_sel = num_seq - num_sel
    for k in msa_feature_list:
        if k in feature:
            if keep_extra and not is_sel:
                new_shape = list(feature[k].shape)
                new_shape[0] = 1
                feature['extra_' + k] = np.zeros(new_shape)
            elif keep_extra and is_sel:
                feature['extra_' + k] = feature[k][not_sel_seq]
            if k == 'msa':
                feature['extra_msa'] = feature['extra_msa'].astype(np.int32)
            feature[k] = feature[k][sel_seq]
    if seed is not None:
        np.random.seed()
    return feature


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


@curry1
def msa_bert_mask(feature=None, uniform_prob=None, profile_prob=None, same_prob=None,
                  replace_fraction=None, seed=None):
    """create masked msa for BERT on raw MSA features"""
    if seed is not None:
        np.random.seed(seed)
    random_aatype = np.array([0.05] * 20 + [0., 0.], dtype=np.float32)

    probability = uniform_prob * random_aatype + profile_prob * feature['hhblits_profile']\
        + same_prob * one_hot(22, feature['msa'])

    pad_shapes = [[0, 0] for _ in range(len(probability.shape))]
    pad_shapes[-1][1] = 1
    mask_prob = 1. - profile_prob - same_prob - uniform_prob

    probability = np.pad(probability, pad_shapes, constant_values=(mask_prob,))

    masked_aatype = np.random.uniform(size=feature['msa'].shape, low=0, high=1) < replace_fraction

    bert_msa = shaped_categorical(probability)
    bert_msa = np.where(masked_aatype, bert_msa, feature['msa'])

    feature['bert_mask'] = masked_aatype.astype(np.int32)
    feature['true_msa'] = feature['msa']
    feature['msa'] = bert_msa
    if seed is not None:
        np.random.seed()
    return feature


def make_atom14_mask(feature):
    "make_atom14_mask"
    feature['atom14_atom_exists'], feature['residx_atom14_to_atom37'],\
        feature['residx_atom37_to_atom14'], feature['atom37_atom_exists'] = \
        make_atom14_masks(feature['aatype'])
    return feature


@curry1
def msa_nearest_neighbor_clusters(feature=None, gap_agreement_weight=0.):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

    # Determine how much weight we assign to each agreement.  In theory, we could
    # use a full blosum matrix here, but right now let's just down-weight gap
    # agreement because it could be spurious.
    # Never put weight on agreeing on BERT mask
    weights = np.concatenate([np.ones(21), gap_agreement_weight * np.ones(1), np.zeros(1)], 0)

    # Make agreement score as weighted Hamming distance
    sample_one_hot = feature["msa_mask"][:, :, None] * one_hot(23, feature['msa'])
    num_seq, num_res, _ = sample_one_hot.shape

    array_extra_msa_mask = feature['extra_msa_mask']
    if array_extra_msa_mask.any():
        extra_one_hot = feature['extra_msa_mask'][:, :, None] * one_hot(23, feature['extra_msa'])
        extra_num_seq, _, _ = extra_one_hot.shape

        agreement = np.matmul(
            np.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
            np.reshape(sample_one_hot * weights, [num_seq, num_res * 23]).T)
        # Assign each sequence in the extra sequences to the closest MSA sample
        feature['extra_cluster_assignment'] = np.argmax(agreement, axis=1)
    else:
        feature['extra_cluster_assignment'] = np.array([])
    return feature


def msa_summarize_clusters(feature=None):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = feature['msa'].shape[0]

    def csum(x):
        result = []
        for i in range(num_seq):
            result.append(np.sum(x[np.where(feature['extra_cluster_assignment'] == i)], axis=0))
        return np.array(result)

    mask = feature['extra_msa_mask']
    mask_counts = 1e-6 + feature['msa_mask'] + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * one_hot(23, feature['extra_msa']))
    msa_sum += one_hot(23, feature['msa'])  # Original sequence
    feature['cluster_profile'] = msa_sum / mask_counts[:, :, None]

    del msa_sum

    del_sum = csum(mask * feature['extra_deletion_matrix'])
    del_sum += feature['deletion_matrix']  # Original sequence
    feature['cluster_deletion_mean'] = del_sum / mask_counts
    del del_sum
    return feature


def msa_feature_concatenate(feature=None):
    """Create and concatenate MSA features."""
    # Whether there is a domain break. Always zero for chains, but keeping
    # for compatibility with domain datasets.
    has_break = np.clip(feature['between_segment_residues'].astype(np.float32),\
                        np.array(0), np.array(1))
    aatype_1hot = one_hot(21, feature['aatype'])

    target_feat = [np.expand_dims(has_break, axis=-1), aatype_1hot]

    msa_1hot = one_hot(23, feature['msa'])
    has_deletion = np.clip(feature['deletion_matrix'], np.array(0), np.array(1))
    deletion_value = np.arctan(feature['deletion_matrix'] / 3.) * (2. / np.pi)

    msa_feat = [msa_1hot, np.expand_dims(has_deletion, axis=-1),\
                np.expand_dims(deletion_value, axis=-1)]

    if feature['cluster_profile'] is not None:
        deletion_mean_value = (np.arctan(feature['cluster_deletion_mean'] / 3.) * (2. / np.pi))
        msa_feat.extend([feature['cluster_profile'], np.expand_dims(deletion_mean_value, axis=-1)])
    feature['extra_has_deletion'] = None
    feature['extra_deletion_value'] = None
    if feature['extra_deletion_matrix'] is not None:
        feature['extra_has_deletion'] = np.clip(feature['extra_deletion_matrix'],
                                                np.array(0),
                                                np.array(1))
        feature['extra_deletion_value'] = np.arctan(feature['extra_deletion_matrix'] / 3.)\
            * (2. / np.pi)

    feature['msa_feat'] = np.concatenate(msa_feat, axis=-1)
    feature['target_feat'] = np.concatenate(target_feat, axis=-1)
    return feature


@curry1
def extra_msa_crop(feature=None, feature_list=None, max_extra_msa=None):
    """MSA features are cropped so only `max_extra_msa` sequences are kept."""
    if feature['extra_msa'].any():
        num_seq = feature['extra_msa'].shape[0]
        num_sel = np.minimum(max_extra_msa, num_seq)
        shuffled = list(range(num_seq))
        np.random.shuffle(shuffled)
        select_indices = shuffled[:num_sel]
    else:
        select_indices = None
    if select_indices:
        feature = dict_take(feature_list, select_indices)(feature)
    return feature


def pseudo_beta_fn(aatype=None, all_atom_positions=None, all_atom_masks=None):
    """compute pseudo beta features from atom positions"""
    is_gly = np.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = np.where(
        np.tile(is_gly[..., None].astype("int32"), [1,] * len(is_gly.shape) + [3,]).astype("bool"),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(is_gly,
                                    all_atom_masks[..., ca_idx],
                                    all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def template_pseudo_beta(feature):
    "template_pseudo_beta"
    feature['template_pseudo_beta'], feature['template_pseudo_beta_mask'] \
        = pseudo_beta_fn(feature['template_aatype'],
                         feature['template_all_atom_positions'],
                         feature['template_all_atom_masks'])
    return feature


@curry1
def template_feature_crop(feature=None, max_templates=None):
    "template_feature_crop"
    for k, v in feature.items():
        if k.startswith('template_'):
            feature[k] = v[:max_templates]
    return feature


def initial_template_mask(feature=None):
    "initial_template_mask"
    if "template_domain_names" in feature:
        num_template = len(feature["template_domain_names"])
    else:
        num_template = 0
    feature['template_mask'] = np.ones([num_template], np.float32)
    feature['msa_mask'] = np.ones(feature['msa'].shape, dtype=np.float32)

    feature['seq_length'] = feature['seq_length'][0]
    return feature


def initial_hhblits_profile(feature=None):
    "initial_hhblits_profile"
    feature['seq_mask'] = np.ones(feature['aatype'].shape, dtype=np.float32)
    if 'hhblits_profile' not in feature:
        # Compute the profile for every residue (over all MSA sequences).
        feature['hhblits_profile'] = np.mean(one_hot(22, feature['msa']), axis=0)
    return feature

MS_MIN32 = -2147483648
MS_MAX32 = 2147483647


def make_random_seed(size, seed_maker_t, low=MS_MIN32, high=MS_MAX32, random_recycle=False):
    "make_random_seed"
    if random_recycle:
        r = np.random.RandomState(seed_maker_t)
        return r.uniform(size=size, low=low, high=high)
    np.random.seed(seed_maker_t)
    return np.random.uniform(size=size, low=low, high=high)


@curry1
def random_crop_to_size(feature=None, feature_list=None, crop_size=None, max_templates=None,
                        max_msa_clusters=None, max_extra_msa=None, subsample_templates=None,
                        seed=None, random_recycle=None):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    seq_length = feature['seq_length']
    seq_length_int = int(seq_length)
    if feature['template_mask'] is not None:
        num_templates = np.array(feature['template_mask'].shape[0], np.int32)
    else:
        num_templates = np.array(0, np.int32)
    num_res_crop_size = np.minimum(seq_length, crop_size)
    num_res_crop_size_int = int(num_res_crop_size)

    # Ensures that the cropping of residues and templates happens in the same way
    # across ensembling iterations.
    # Do not use for randomness that should vary in ensembling.
    if subsample_templates:
        templates_crop_start = int(make_random_seed(size=(),
                                                    seed_maker_t=seed,
                                                    low=0,
                                                    high=num_templates + 1,
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

    for k, v in feature.items():
        if k not in feature_list or ('template' not in k and NUM_RES not in feature_list.get(k)):
            continue
        # randomly permute the templates before cropping them.
        if k.startswith('template') and subsample_templates:
            v = v[templates_select_indices]

        crop_sizes = []
        crop_starts = []
        for i, (dim_size, dim) in enumerate(zip(feature_list.get(k), v.shape)):
            is_num_res = dim_size == NUM_RES
            if i == 0 and k.startswith('template'):
                crop_size_ = num_templates_crop_size_int
                crop_start = templates_crop_start
            else:
                crop_start = num_res_crop_start if is_num_res else 0
                crop_size_ = (num_res_crop_size_int if is_num_res else (-1 if dim is None else dim))
            crop_sizes.append(crop_size_)
            crop_starts.append(crop_start)
        if len(v.shape) == 1:
            feature[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0]]
        if len(v.shape) == 2:
            feature[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                           crop_starts[1]:crop_starts[1] + crop_sizes[1]]
        if len(v.shape) == 3:
            feature[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                           crop_starts[1]:crop_starts[1] + crop_sizes[1],
                           crop_starts[2]:crop_starts[2] + crop_sizes[2]]
        if len(v.shape) == 4:
            feature[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                           crop_starts[1]:crop_starts[1] + crop_sizes[1],
                           crop_starts[2]:crop_starts[2] + crop_sizes[2],
                           crop_starts[3]:crop_starts[3] + crop_sizes[3]]

    feature["num_residues"] = feature["seq_length"]
    feature["seq_length"] = num_res_crop_size
    pad_size_map = {
        NUM_RES: crop_size,
        NUM_MSA_SEQ: max_msa_clusters,
        NUM_EXTRA_SEQ: max_extra_msa,
        NUM_TEMPLATES: max_templates,
    }

    for k, v in feature.items():
        if k not in feature_list or k == "num_residues":
            continue
        shape = list(v.shape)
        schema = feature_list.get(k)
        assert len(shape) == len(
            schema), f'Rank mismatch between shape and shape schema for {k}: {shape} vs {schema}'

        pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        if padding:
            feature[k] = np.pad(v, padding)
            feature[k].reshape(pad_size)

    return feature


def label_make_atom14_positions(label=None):
    "label_make_atom14_positions"
    label["atom14_atom_exists"], label["atom14_gt_exists"], label["atom14_gt_positions"],\
        label["residx_atom14_to_atom37"], label["residx_atom37_to_atom14"],\
        label["atom37_atom_exists"], label["atom14_alt_gt_positions"],\
        label["atom14_alt_gt_exists"], label["atom14_atom_is_ambiguous"]\
             = make_atom14_positions(label["aatype"],
                                     label["all_atom_mask"],
                                     label["all_atom_positions"])
    return label


def gather(params=None, indices=None, axis=0):
    """gather operation"""
    func = lambda p, i: np.take(p, i, axis=axis)
    return func(params, indices)


def np_gather_ops(params=None, indices=None, axis=0, batch_dims=0):
    """np gather operation"""
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

    k = [[xx + yy + zz, zy - yz, xz - zx, yx - xy,],
         [zy - yz, xx - yy - zz, xy + yx, xz + zx,],
         [xz - zx, xy + yx, yy - xx - zz, yz + zy,],
         [yx - xy, xz + zx, yz + zy, zz - xx - yy,]]

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


@curry1
def label_atom37_to_frames(label=None, is_affine=False):
    """get the frames and affine for each residue"""
    aatype_shape = label['aatype'].shape

    flat_aatype = np.reshape(label['aatype'], [-1])
    all_atom_positions = np.reshape(label['all_atom_positions'], [-1, 37, 3])
    all_atom_mask = np.reshape(label['all_atom_mask'], [-1, 37])

    rigid_group_names_res = np.full([21, 8, 3], '', dtype=object)

    # group 0: backbone frame
    rigid_group_names_res[:, 0, :] = ['C', 'CA', 'N']

    # group 3: 'psi'
    rigid_group_names_res[:, 3, :] = ['CA', 'C', 'O']

    # group 4,5,6,7: 'chi1,2,3,4'
    for restype, letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3.get(letter, -1)
        for chi_idx in range(4):
            if residue_constants.chi_angles_mask[restype][chi_idx]:
                atom_names = residue_constants.chi_angles_atoms[restype_name][chi_idx]
                rigid_group_names_res[restype, chi_idx + 4, :] = atom_names[1:]

    # create rigid group mask
    rigid_group_mask_res = np.zeros([21, 8], dtype=np.float32)
    rigid_group_mask_res[:, 0] = 1
    rigid_group_mask_res[:, 3] = 1
    rigid_group_mask_res[:20, 4:] = residue_constants.chi_angles_mask

    lookup_table = residue_constants.atom_order.copy()
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
    gt_frames = geometry.rigids_mul_rots(gt_frames,
                                         geometry.rots_from_tensor(rotations, use_numpy=True))

    rigid_group_is_ambiguous_res = np.zeros([21, 8], dtype=np.float32)
    rigid_group_rotations_res = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])

    for restype_name, _ in residue_constants.residue_atom_renaming_swaps.items():
        restype = residue_constants.restype_order[residue_constants.restype_3to1[restype_name]]
        chi_idx = int(sum(residue_constants.chi_angles_mask[restype]) - 1)
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
        gt_frames, geometry.rots_from_tensor(rigid_group_ambiguity_rotation_res_index,
                                             use_numpy=True))

    gt_frames_flat12 = np.stack(list(gt_frames[0]) + list(gt_frames[1]), axis=-1)
    alt_gt_frames_flat12 = np.stack(list(alt_gt_frames[0]) + list(alt_gt_frames[1]), axis=-1)
    # reshape back to original residue layout
    gt_frames_flat12 = np.reshape(gt_frames_flat12, aatype_shape + (8, 12))
    gt_masks = np.reshape(gt_masks, aatype_shape + (8,))
    group_masks = np.reshape(group_masks, aatype_shape + (8,))
    gt_frames_flat12 = np.reshape(gt_frames_flat12, aatype_shape + (8, 12))
    rigid_group_is_ambiguous_res_index = np.reshape(rigid_group_is_ambiguous_res_index,
                                                    aatype_shape + (8,))
    alt_gt_frames_flat12 = np.reshape(alt_gt_frames_flat12,
                                      aatype_shape + (8, 12,))
    # if not is_affine:
    label['rigidgroups_gt_frames'] = gt_frames_flat12  # shape (..., 8, 12)
    label['rigidgroups_gt_exists'] = gt_masks  # shape (..., 8)
    label['rigidgroups_group_exists'] = group_masks  # shape (..., 8)
    label['rigidgroups_group_is_ambiguous'] = rigid_group_is_ambiguous_res_index  # shape (..., 8)
    label['rigidgroups_alt_gt_frames'] = alt_gt_frames_flat12  # shape (..., 8, 12)
    if is_affine:
        rotation = [[gt_frames[0][0], gt_frames[0][1], gt_frames[0][2]],
                    [gt_frames[0][3], gt_frames[0][4], gt_frames[0][5]],
                    [gt_frames[0][6], gt_frames[0][7], gt_frames[0][8]]]
        translation = [gt_frames[1][0], gt_frames[1][1], gt_frames[1][2]]
        backbone_affine_tensor = to_tensor(rotation, translation)[:, 0, :]
        label['backbone_affine_tensor'] = backbone_affine_tensor
    return label


def get_chi_atom_pos_indices():
    """get the atom indices for computing chi angles for all residue types"""
    chi_atom_pos_indices = []
    for residue_name in residue_constants.restypes:
        residue_name = residue_constants.restype_1to3[residue_name]
        residue_chi_angles = residue_constants.chi_angles_atoms[residue_name]
        atom_pos_indices = []
        for chi_angle in residue_chi_angles:
            atom_pos_indices.append([residue_constants.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_pos_indices)):
            atom_pos_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_pos_indices.append(atom_pos_indices)

    chi_atom_pos_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.array(chi_atom_pos_indices)


@curry1
def label_atom37_to_torsion_angles(label=None, alt_torsions=False):
    """get the torsion angles of each residue"""
    aatype = label['aatype'].reshape((1, -1))
    all_atom_pos = label['all_atom_positions'].reshape((1, -1, 37, 3))
    all_atom_mask = label['all_atom_mask'].reshape((1, -1, 37))

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
    atom_pos_indices = np_gather_ops(chi_atom_pos_indices, true_aatype, 0, 0)
    chi_atom_pos = np_gather_ops(all_atom_pos, atom_pos_indices, -2, 2)

    angles_mask = list(residue_constants.chi_angles_mask)
    angles_mask.append([0.0, 0.0, 0.0, 0.0])
    angles_mask = np.array(angles_mask)

    chis_mask = np_gather_ops(angles_mask, true_aatype, 0, 0)

    chi_angle_atoms_mask = np_gather_ops(all_atom_mask, atom_pos_indices, -1, 2)

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

    torsion_angles_sin_cos *= np.array(
        [1., 1., -1., 1., 1., 1., 1.])[None, None, :, None]

    chi_is_ambiguous = np_gather_ops(
        np.array(residue_constants.chi_pi_periodic), true_aatype)
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
    label['torsion_angles_sin_cos'] = torsion_angles_sin_cos[0]  # (N, 7, 2)
    label['alt_torsion_angles_sin_cos'] = alt_torsion_angles_sin_cos[0]  # (N, 7, 2)
    label['torsion_angles_mask'] = torsion_angles_mask_padding[0]  # (N, 7)
    return label


def label_pseudo_beta(label=None):
    "label_pseudo_beta"
    label['pseudo_beta'], label['pseudo_beta_mask'] = pseudo_beta_fn(label['aatype'],
                                                                     label['all_atom_positions'],
                                                                     label['all_atom_mask'])
    label["chi_mask"] = label.get("torsion_angles_mask")[:, 3:]
    label['torsion_angles_sin_cos'] = label.get('torsion_angles_sin_cos')[:, 3:, :]
    label['alt_torsion_angles_sin_cos'] = label.get('alt_torsion_angles_sin_cos')[:, 3:, :]
    label['backbone_affine_mask'] = label['pseudo_beta_mask']
    label.pop("aatype")
    return label


def prev_initial(feature=None):
    "prev_initial"
    feature['prev_pos'] = np.zeros([feature['aatype'].shape[1], 37, 3])
    feature['prev_msa_first_row'] = np.zeros([feature['aatype'].shape[1], 256])
    feature['prev_pair'] = np.zeros([feature['aatype'].shape[1], feature['aatype'].shape[1], 128])
    return feature


def tail_process(feature=None):
    "tail_process"
    feature["atomtype_radius"] = np.array(
        [1.55, 1.7, 1.7, 1.7, 1.52, 1.7, 1.7, 1.7, 1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.55, 1.55,
         1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.7, 1.55, 1.55, 1.55, 1.52, 1.52, 1.7, 1.55, 1.55,
         1.52, 1.7, 1.7, 1.7, 1.55, 1.52])
    feature["restype_atom14_bond_lower_bound"], feature["restype_atom14_bond_upper_bound"], _ = \
            residue_constants.make_atom14_dists_bounds(overlap_tolerance=1.5,
                                                       bond_length_tolerance_factor=12.0)
    feature["use_clamped_fape"] = np.random.binomial(1, 0.9, size=1)
    feature["filter_by_solution"] = np.array(1.0)
    return feature
