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
"""multimer data"""
import numpy as np
from ....common import residue_constants
from ...dataset import curry1
from ....data.data_transform import make_atom14_masks
from .multimer_feature import NUM_RES, NUM_MSA_SEQ, NUM_EXTRA_SEQ, NUM_TEMPLATES


@curry1
def dict_filter_key(feature=None, feature_list=None):
    feature = {k: v for k, v in feature.items() if k in feature_list}
    return feature


@curry1
def dict_replace_key(feature=None, replaced_key=None):
    assert len(replaced_key) == 2
    origin_key, new_key = replaced_key
    if origin_key in feature:
        feature[new_key] = feature.pop(origin_key)
    return feature


@curry1
def dict_cast(feature=None, cast_type=None, filtered_list=None):
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
    for k in filter_list:
        if k in feature:
            feat_dim = feature[k].shape[axis]
            if isinstance(feat_dim, int) and feat_dim == 1:
                feature[k] = np.squeeze(feature[k], axis=axis)
    return feature


@curry1
def dict_take(feature=None, filter_list=None, axis=None):
    for k in filter_list:
        if k in feature:
            feature[k] = feature[k][axis]
    return feature


@curry1
def dict_del_key(feature=None, filter_list=None):
    for k in filter_list:
        if k in feature:
            del feature[k]
    return feature


@curry1
def one_hot_convert(feature=None, key=None, axis=None):
    if key in feature:
        feature[key] = np.argmax(feature[key], axis=axis)
    return feature


@curry1
def correct_restypes(feature=None, key=None):
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, dtype=feature[key].dtype)
    feature[key] = new_order[feature[key]]
    return feature


@curry1
def make_msa_profile(feature=None, axis=None, drop_mask_channel=False, eps=1e-10):
    """Make_msa_profile."""
    mask = feature['msa_mask'][:, :, None]
    value = np.eye(22)[feature['msa']]
    feature['target_feat'] = np.eye(21)[feature['aatype']]
    if drop_mask_channel:
        mask = mask[..., 0]
    mask_shape = mask.shape
    value_shape = value.shape
    broadcast_factor = 1.
    value_size = value_shape[axis]
    mask_size = mask_shape[axis]
    if mask_size == 1:
        broadcast_factor *= value_size
    feature['msa_profile'] = np.sum(mask * value, axis=axis) / (np.sum(mask, axis=axis) * broadcast_factor + eps)
    return feature


@curry1
def sample_msa(feature=None, msa_feature_list=None, max_seq=None, seed=None):
    """Sample MSA randomly."""
    if seed is not None:
        np.random.seed(seed)

    logits = (np.clip(np.sum(feature['msa_mask'], axis=-1), 0., 1.) - 1.) * 1e6
    if 'cluster_bias_mask' not in feature:
        cluster_bias_mask = np.pad(
            np.zeros(feature['msa'].shape[0] - 1), (1, 0), constant_values=1.)
    else:
        cluster_bias_mask = feature['cluster_bias_mask']
    logits += cluster_bias_mask * 1e6
    z = np.random.gumbel(loc=0.0, scale=1.0, size=logits.shape)
    index_order = np.argsort((logits + z), axis=-1, kind='quicksort', order=None)
    sel_idx = index_order[:max_seq]
    extra_idx = index_order[max_seq:]
    for k in msa_feature_list:
        if k in feature:
            feature['extra_' + k] = feature[k][extra_idx]
            feature[k] = feature[k][sel_idx]

    if seed is not None:
        np.random.seed()
    return feature


@curry1
def make_masked_msa(feature=None, config=None, epsilon=1e-6, seed=None):
    """create data for BERT on raw MSA."""
    if seed is not None:
        np.random.seed(seed)

    random_aa = np.array([0.05] * 20 + [0., 0.], dtype=np.float32)
    categorical_probs = (
        config.uniform_prob * random_aa +
        config.profile_prob * feature['msa_profile'] +
        config.same_prob * np.eye(22)[feature['msa']])
    pad_shapes = [[0, 0] for _ in range(len(categorical_probs.shape))]
    pad_shapes[-1][1] = 1
    mask_prob = 1. - config.profile_prob - config.same_prob - config.uniform_prob
    categorical_probs = np.pad(categorical_probs, pad_shapes, constant_values=mask_prob)
    sh = feature['msa'].shape
    mask_position = (np.random.uniform(0., 1., sh) < config.replace_fraction).astype(np.float32)
    mask_position *= feature['msa_mask']
    logits = np.log(categorical_probs + epsilon)
    z = np.random.gumbel(loc=0.0, scale=1.0, size=logits.shape)
    bert_msa = np.eye(logits.shape[-1], dtype=logits.dtype)[np.argmax(logits + z, axis=-1)]
    bert_msa = (np.where(mask_position,
                         np.argmax(bert_msa, axis=-1), feature['msa']))
    bert_msa *= (feature['msa_mask'].astype(np.int64))
    if 'bert_mask' in feature:
        feature['bert_mask'] *= mask_position.astype(np.float32)
    else:
        feature['bert_mask'] = mask_position.astype(jnp.float32)
    feature['true_msa'] = feature['msa']
    feature['msa'] = bert_msa

    if seed is not None:
        np.random.seed()
    return feature


def softmax(x, axis):
    """ Softmax func"""
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x


def nearest_neighbor_clusters(feature, gap_agreement_weight=0., seed=None):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""
    if seed is not None:
        np.random.seed(seed)

    weights = np.array(
        [1.] * 21 + [gap_agreement_weight] + [0.], dtype=np.float32)
    msa_mask = feature['msa_mask']
    msa_one_hot = np.eye(23)[feature['msa']]
    extra_mask = feature['extra_msa_mask']
    extra_one_hot = np.eye(23)[feature['extra_msa']]
    msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
    extra_one_hot_masked = extra_mask[:, :, None] * extra_one_hot
    agreement = np.einsum('mrc, nrc->nm', extra_one_hot_masked,
                          weights * msa_one_hot_masked)
    cluster_assignment = softmax(1e3 * agreement, axis=0)
    cluster_assignment *= np.einsum('mr, nr->mn', msa_mask, extra_mask)
    cluster_count = np.sum(cluster_assignment, axis=-1)
    cluster_count += 1.
    msa_sum = np.einsum('nm, mrc->nrc', cluster_assignment, extra_one_hot_masked)
    msa_sum += msa_one_hot_masked
    feature['cluster_profile'] = msa_sum / cluster_count[:, None, None]
    extra_deletion_matrix = feature['extra_deletion_matrix']
    deletion_matrix = feature['deletion_matrix']
    del_sum = np.einsum('nm, mc->nc', cluster_assignment,
                        extra_mask * extra_deletion_matrix)
    del_sum += deletion_matrix
    feature['cluster_deletion_mean'] = del_sum / cluster_count[:, None]

    if seed is not None:
        np.random.seed()
    return feature


def create_msa_feat(feature):
    """Create and concatenate MSA features."""
    msa_1hot = np.eye(23)[feature['msa']]
    deletion_matrix = feature['deletion_matrix']
    has_deletion = np.clip(deletion_matrix, 0., 1.)[..., None]
    deletion_value = (np.arctan(deletion_matrix / 3.) * (2. / np.pi))[..., None]
    deletion_mean_value = (np.arctan(feature['cluster_deletion_mean'] / 3.) *
                           (2. / np.pi))[..., None]
    msa_feat = [
        msa_1hot,
        has_deletion,
        deletion_value,
        feature['cluster_profile'],
        deletion_mean_value
    ]
    feature['msa_feat'] = np.concatenate(msa_feat, axis=-1)
    return feature


def make_atom14_mask(feature):
    _, _, feature['residx_atom37_to_atom14'], feature['atom37_atom_exists'] = \
        make_atom14_masks(feature['aatype'])
    return feature


MS_MIN32 = -2147483648
MS_MAX32 = 2147483647


def make_random_seed(size, seed_maker_t, low=MS_MIN32, high=MS_MAX32, random_recycle=False):
    if random_recycle:
        r = np.random.RandomState(seed_maker_t)
        return r.uniform(size=size, low=low, high=high)
    np.random.seed(seed_maker_t)
    return np.random.uniform(size=size, low=low, high=high)


@curry1
def random_crop_to_size(feature=None, feature_list=None, crop_size=None, max_templates=None, max_msa_clusters=None,
                        max_extra_msa=None, seed=None, random_recycle=None):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    seq_length = feature['seq_length']
    seq_length_int = int(seq_length)
    num_templates = np.array(0, np.int32)
    num_res_crop_size = np.minimum(seq_length, crop_size)
    num_res_crop_size_int = int(num_res_crop_size)

    # Ensures that the cropping of residues and templates happens in the same way
    # across ensembling iterations.
    # Do not use for randomness that should vary in ensembling.
    templates_crop_start = 0
    num_templates_crop_size = np.minimum(num_templates - templates_crop_start, max_templates)
    num_templates_crop_size_int = int(num_templates_crop_size)

    num_res_crop_start = int(make_random_seed(size=(), seed_maker_t=seed, low=0,
                                              high=seq_length_int - num_res_crop_size_int + 1,
                                              random_recycle=random_recycle))

    for k, v in feature.items():
        if k not in feature_list or ('template' not in k and NUM_RES not in feature_list.get(k)):
            continue

        crop_sizes = []
        crop_starts = []
        for i, (dim_size, dim) in enumerate(zip(feature_list.get(k), v.shape)):
            is_num_res = (dim_size == NUM_RES)
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
        elif len(v.shape) == 2:
            feature[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                           crop_starts[1]:crop_starts[1] + crop_sizes[1]]
        elif len(v.shape) == 3:
            feature[k] = v[crop_starts[0]:crop_starts[0] + crop_sizes[0],
                           crop_starts[1]:crop_starts[1] + crop_sizes[1],
                           crop_starts[2]:crop_starts[2] + crop_sizes[2]]
        else:
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


def prev_initial(feature):
    feature['prev_pos'] = np.zeros([feature['aatype'].shape[1], 37, 3])
    feature['prev_msa_first_row'] = np.zeros([feature['aatype'].shape[1], 256])
    feature['prev_pair'] = np.zeros([feature['aatype'].shape[1], feature['aatype'].shape[1], 128])
    return feature
