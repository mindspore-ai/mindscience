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
"""rasp_data"""

import numpy as np
from ...dataset import curry1

def make_contact_info(ori_seq_len, ur_path):
    '''make_contact_info'''
    num_residues = ori_seq_len
    contact_info_mask = np.zeros((num_residues, num_residues))
    if not ur_path:
        return contact_info_mask
    with open(ur_path, encoding='utf-8') as f:
        all_urs = f.readlines()
    all_urs = [i.split('!')[0].rstrip() for i in all_urs]
    useful_urs = []
    for urls in all_urs:
        i = urls.split(" ")
        temp = []
        temp.append(int(i[0]))
        temp.append(int(i[-1]))
        useful_urs.append(temp)

    for i in useful_urs:
        contact_info_mask[i[0], i[1]] = 1
    contact_info_mask = (contact_info_mask + contact_info_mask.T) > 0
    contact_info_mask = contact_info_mask.astype(np.float32)

    return contact_info_mask


@curry1
def contact_info(feature=None, contact_path=None):
    "contact_info"
    feature["contact_info_mask"] = make_contact_info(256, contact_path)
    print(feature["contact_info_mask"].shape)
    return feature
