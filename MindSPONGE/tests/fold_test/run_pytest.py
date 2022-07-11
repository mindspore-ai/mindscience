# Copyright 2021 Huawei Technologies Co., Ltd
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
"""run script"""

import time
import pickle
import pytest
import numpy as np

import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore import load_checkpoint

from data.feature.feature_extraction import process_features
from commons.utils import compute_confidence
from model import AlphaFold
from config import config, global_config

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    variable_memory_max_size="31GB",
                    device_id=0,
                    save_graphs=False)
model_name = "model_1"
model_config = config.model_config(model_name)

global_config = global_config.global_config(1024)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_fold():
    """test case for fold inference"""
    num_recycle = model_config.model.num_recycle
    fold_net = AlphaFold(model_config, global_config)
    load_checkpoint("./protein_fold_1.ckpt", fold_net)

    t1 = time.time()
    with open(f'./features24.pkl', 'rb') as f:
        input_features = pickle.load(f)
    tensors, aatype, residue_index, ori_res_length = process_features(
        raw_features=input_features, config=model_config, global_config=global_config)
    prev_pos = Tensor(np.zeros([global_config.seq_length, 37, 3]).astype(np.float16))
    prev_msa_first_row = Tensor(np.zeros([global_config.seq_length, 256]).astype(np.float16))
    prev_pair = Tensor(np.zeros([global_config.seq_length, global_config.seq_length, 128]).astype(np.float16))

    for i in range(num_recycle+1):
        tensors_i = [tensor[i] for tensor in tensors]
        input_feats = [Tensor(tensor) for tensor in tensors_i]
        final_atom_positions, final_atom_mask, predicted_lddt_logits,\
            prev_pos, prev_msa_first_row, prev_pair = fold_net(*input_feats,
                                                               prev_pos,
                                                               prev_msa_first_row,
                                                               prev_pair)

    final_atom_positions = final_atom_positions.asnumpy()[:ori_res_length]
    final_atom_mask = final_atom_mask.asnumpy()[:ori_res_length]
    predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]

    confidence = compute_confidence(predicted_lddt_logits)

    t4 = time.time()
    right_result = np.load("result24.npy")
    l1_error = np.max(np.abs(final_atom_positions - right_result))
    l1_error = l1_error + np.sum(final_atom_mask) * 0 \
        + np.sum(aatype) * 0 + np.sum(residue_index) * 0

    print('test_res:', f'l1_error: {l1_error:.10f} ')
    print(f'confidence===: {confidence[0]:.10f} ')
    print(f'total time===: {t4 - t1:.10f} ')
    assert l1_error <= 5
    assert (t4 - t1) <= 1800
    assert confidence[0] >= 90
