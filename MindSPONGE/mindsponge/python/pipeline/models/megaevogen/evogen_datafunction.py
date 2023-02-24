# Copyright 2023 Huawei Technologies Co., Ltd
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
"""evogen"""
import numpy as np
from ...dataset import curry1
from ....common import residue_constants
from ....data.data_transform import one_hot

MS_MIN32 = -2147483648
MS_MAX32 = 2147483647
NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'
NUM_SEQ = "length msa placeholder"
NUM_NOISE = 'num noise placeholder'
NUM_LATENT_DIM = "num latent placeholder"


@curry1
def dict_filter_key(feature=None, feature_list=None):
    '''dict_filter_key'''
    feature = {k: v for k, v in feature.items() if k in feature_list}
    return feature


@curry1
def dict_replace_key(feature=None, replaced_key=None):
    '''dict_replace_key'''
    assert len(replaced_key) == 2
    origin_key, new_key = replaced_key
    if origin_key in feature:
        feature[new_key] = feature.pop(origin_key)
    return feature


@curry1
def dict_cast(feature=None, cast_type=None, filtered_list=None):
    '''dict_cast'''
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
    '''dict_suqeeze'''
    for k in filter_list:
        if k in feature:
            dim = feature[k].shape[axis]
            if isinstance(dim, int) and dim == 1:
                feature[k] = np.squeeze(feature[k], axis=axis)
    return feature


@curry1
def dict_concatenate(feature=None, keys=None, result_key=None, axis=-1):
    '''dict_concatenate'''
    to_concatenate_data = []
    for key in keys:
        if key in feature:
            to_concatenate_data.append(feature[key])
    data = np.concatenate(to_concatenate_data, axis=axis)
    feature[result_key] = data
    return feature


@curry1
def dict_expand_dims(feature=None, keys=None, result_key=None, axis=-1):
    '''dict_expand_dims'''
    if result_key:
        assert len(result_key) == len(keys)
        key_num = len(keys)
        for i in range(key_num):
            feature[result_key[i]] = np.expand_dims(feature[keys[i]], axis=axis)
    else:
        for key in range(keys):
            feature[key] = np.expand_dims(feature[key], axis=axis)
    return feature


@curry1
def dict_del_key(feature=None, filter_list=None):
    '''dict_del_key'''
    for k in filter_list:
        if k in feature:
            del feature[k]
    return feature


@curry1
def dict_take(feature=None, filter_list=None, result_key=None, axis=0):
    '''dict_take'''
    if result_key:
        assert len(filter_list) == len(result_key)
        for i, item in enumerate(filter_list):
            if item in feature:
                feature[result_key[i]] = feature[item][axis]
    else:
        for k in filter_list:
            if k in feature:
                feature[k] = feature[k][axis]
    return feature


@curry1
def one_hot_convert(feature=None, key=None, axis=0):
    '''one_hot_convert'''
    if key in feature:
        feature[key] = np.argmax(feature[key], axis=axis)
    return feature


@curry1
def correct_restypes(feature=None, key=None):
    '''correct_restypes'''
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, dtype=feature[key].dtype)
    feature[key] = new_order[feature[key]]
    return feature


@curry1
def make_mask(feature=None, key=None, result_key=None):
    '''make_mask'''
    feature[result_key] = np.ones(feature[key].shape, dtype=np.float32)
    return feature


def initialize_hhblits_profile(feature=None):
    '''initialize_hhblits_profile'''
    if 'hhblits_profile' not in feature:
        feature['hhblits_profile'] = np.mean(one_hot(22, feature['msa']), axis=0)
    return feature


@curry1
def msa_sample(feature=None, msa_feature_list=None, keep_extra=True, max_msa_clusters=128, seed=None):
    '''msa_sample'''
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
            if keep_extra and k == 'msa':
                feature['extra_msa'] = feature['extra_msa'].astype(np.int32)
            feature[k] = feature[k][sel_seq]
    if seed is not None:
        np.random.seed()
    return feature


def shape_list(x=None):
    """get the list of dimensions of an array"""
    x = np.array(x)
    if x.ndim is None:
        return x.shape

    static = x.shape
    ret = []
    for _, dimension in enumerate(static):
        ret.append(dimension)
    return ret


def shaped_categorical(probability=None):
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
def msa_bert_mask(feature=None, uniform_prob=None, profile_prob=None, same_prob=None, replace_fraction=None, seed=None):
    """create masked msa for BERT on raw MSA features"""
    if seed is not None:
        np.random.seed(seed)
    random_aatype = np.array([0.05] * 20 + [0., 0.], dtype=np.float32)

    probability = uniform_prob * random_aatype + profile_prob * feature['hhblits_profile'] + \
                  same_prob * one_hot(22, feature['msa'])

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


@curry1
def generate_random_sample(feature=None, num_noise=None, latent_dim=None, context_true_prob=None,
                           keep_prob=None, available_msa_fraction=None,
                           max_msa_clusters=None, crop_size=None, seed=None):
    '''generate_random_sample'''
    if seed is not None:
        np.random.seed(seed)

    context_true_prob = np.absolute(context_true_prob)
    keep_prob = np.absolute(keep_prob)

    available_msa = int(available_msa_fraction * max_msa_clusters)
    available_msa = min(available_msa, max_msa_clusters)

    feature['evogen_random_data'] = np.random.normal(
        size=(num_noise, max_msa_clusters, crop_size, latent_dim)).astype(np.float32)

    context_mask = np.zeros((max_msa_clusters,), np.int32)
    z1 = np.random.random(max_msa_clusters)
    context_mask = np.asarray([1 if x < context_true_prob else 0 for x in z1], np.int32)
    context_mask[available_msa:] *= 0

    target_mask = np.zeros((max_msa_clusters,), np.int32)
    z2 = np.random.random(max_msa_clusters)
    target_mask = np.asarray([1 if x < keep_prob else 0 for x in z2], np.int32)

    context_mask[0] = 1
    target_mask[0] = 1

    feature['evogen_context_mask'] = np.stack((context_mask, target_mask), -1)
    if seed is not None:
        np.random.seed()
    return feature


def make_random_seed(size=None, seed_maker_t=None, low=MS_MIN32, high=MS_MAX32, random_recycle=False):
    '''make_random_seed'''
    if random_recycle:
        r = np.random.RandomState(seed_maker_t)
        return r.uniform(size=size, low=low, high=high)
    np.random.seed(seed_maker_t)
    return np.random.uniform(size=size, low=low, high=high)


def feature_pad(feature=None, pad_size_map=None, feature_list=None):
    '''feature_pad'''
    for k, v in feature.items():
        if k not in feature_list or k == "seq_length":
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


@curry1
def random_crop_to_size(feature=None, feature_list=None, crop_size=None, max_msa_clusters=None, seed=None,
                        num_templates_crop_size_int=4, templates_crop_start=0):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    seq_length = feature['seq_mask'].shape[0]
    seq_length_int = int(seq_length)

    num_res_crop_size = np.minimum(seq_length, crop_size)
    num_res_crop_size_int = int(num_res_crop_size)

    num_res_crop_start = int(make_random_seed(size=(), seed_maker_t=seed, low=0,
                                              high=seq_length_int - num_res_crop_size_int + 1,
                                              random_recycle=False))

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

    feature["seq_length"] = np.array(seq_length, np.int32)
    pad_size_map = {
        NUM_RES: crop_size,
        NUM_MSA_SEQ: max_msa_clusters
    }
    feature = feature_pad(feature, pad_size_map, feature_list)

    return feature
