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
"""colabdesign data"""
import numpy as np
import mindsponge.common.residue_constants as residue_constants

from ...dataset import curry1
from ....common import residue_constants


@curry1
def dict_filter_key(feature, feature_list):
    feature = {k: v for k, v in feature.items() if k in feature_list}
    return feature


@curry1
def dict_replace_key(feature, replaced_key):
    assert len(replaced_key) == 2
    origin_key, new_key = replaced_key
    if origin_key in feature:
        feature[new_key] = feature.pop(origin_key)
    return feature


@curry1
def dict_cast(feature, cast_type, filtered_list):
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
def dict_take(feature, filter_list, axis):
    for k in filter_list:
        if k in feature:
            feature[k] = feature[k][axis]
    return feature


@curry1
def dict_del_key(feature, filter_list):
    for k in filter_list:
        if k in feature:
            del feature[k]
    return feature


@curry1
def one_hot_convert(feature, key, axis):
    if key in feature:
        feature[key] = np.argmax(feature[key], axis=axis)
    return feature


@curry1
def correct_restypes(feature, key):
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, dtype=feature[key].dtype)
    feature[key] = new_order[feature[key]]
    return feature


@curry1
def prep(feature=None, cfg=None):
    prev_pos = np.zeros((cfg.seq_length, 37, 3)).astype(np.float32)
    prev_msa_first_row = np.zeros((cfg.seq_length, cfg.model.msa_channel)).astype(np.float32)
    prev_pair = np.zeros((cfg.seq_length, cfg.seq_length, cfg.model.pair_channel)).astype(np.float32)
    feature.append(prev_pos)
    feature.append(prev_msa_first_row)
    feature.append(prev_pair)
    return feature


@curry1
def get_weights(feature=None, index=None, cfg=None):
    """get weights"""
    opt_temp = []
    opt_soft = []
    opt_hard = []

    for i in range(cfg.soft_iters):
        opt_temp.append(
            cfg.soft_etemp + (cfg.soft_temp - cfg.soft_etemp) * (1 - (i + 1) / cfg.soft_iters) ** 2)
        opt_soft.append((i + 1) / cfg.soft_iters)
        opt_hard.append(cfg.soft_hard)
    for i in range(cfg.temp_iters):
        opt_temp.append(
            cfg.temp_decay + (cfg.temp_value - cfg.temp_decay) * (1 - (i + 1) / cfg.temp_iters) ** 2)
        opt_soft.append(cfg.temp_esoft + (cfg.temp_soft - cfg.temp_esoft) * ((i + 1) / cfg.temp_iters))
        opt_hard.append(cfg.temp_ehard + (cfg.temp_hard - cfg.temp_ehard) * ((i + 1) / cfg.temp_iters))
    for i in range(cfg.hard_iters):
        opt_temp.append(
            cfg.hard_etemp + (cfg.hard_temp - cfg.hard_etemp) * (1 - (i + 1) / cfg.hard_iters) ** 2)
        opt_soft.append(cfg.hard_esoft + (cfg.hard_soft - cfg.hard_esoft) * ((i + 1) / cfg.hard_iters))
        opt_hard.append(cfg.hard_decay + (cfg.hard_value - cfg.hard_decay) * ((i + 1) / cfg.hard_iters))
    feature.append(opt_temp[index])
    feature.append(opt_soft[index])
    feature.append(opt_hard[index])
    return feature
