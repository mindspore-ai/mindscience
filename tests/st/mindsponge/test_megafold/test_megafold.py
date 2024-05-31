# Copyright 2022 Huawei Technologies Co., Ltd
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
"""eval script"""
import os
import pickle
import pytest
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore import load_checkpoint
from mindsponge.common.config_load import load_config
from mindsponge.cell.amp import amp_convert
from model.fold import MegaFold, compute_confidence
from data.preprocess import Feature
from data.utils import get_raw_feature


def load_pkl(pickle_path):
    f = open(pickle_path, "rb")
    data = pickle.load(f)
    f.close()
    return data


def fold_infer(crop_size, predict_confidence, mixed_precision=False):
    """fold_infer"""
    data_cfg = load_config('./config/data.yaml')
    model_cfg = load_config('./config/model.yaml')
    data_cfg.eval.crop_size = crop_size
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=mixed_precision)
    if mixed_precision is True:
        fp32_white_list = (nn.Softmax, nn.LayerNorm)
        amp_convert(megafold, fp32_white_list)
    else:
        megafold.to_float(mstype.float32)
    load_checkpoint("/home/workspace/mindspore_ckpt/ckpt/MEGA_Fold_1.ckpt", megafold)
    seq_files = os.listdir("./feature/")
    feature_generator = None
    for seq_file in seq_files:
        raw_feature = get_raw_feature(os.path.join("./feature/", seq_file), feature_generator, True)
        ori_res_length = raw_feature['msa'].shape[1]
        processed_feature = Feature(data_cfg, raw_feature)
        feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg,
                                                                                   mixed_precision=mixed_precision)
        prev_pos = Tensor(prev_pos)
        prev_msa_first_row = Tensor(prev_msa_first_row)
        prev_pair = Tensor(prev_pair)
        for i in range(data_cfg.common.num_recycle):
            feat_i = [Tensor(x[i]) for x in feat]
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = megafold(*feat_i,
                                                                                      prev_pos,
                                                                                      prev_msa_first_row,
                                                                                      prev_pair)
        predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]
        confidence = compute_confidence(predicted_lddt_logits, return_lddt=False)
        print("confidence:", confidence)
        assert confidence > predict_confidence

@pytest.mark.level0
@pytest.mark.platfrom_arm_ascend910b_training
@pytest.mark.env_onecard
def test_megafold_ascend_seqlen_256():
    """
    Feature: megafold model test in the ascend, seq length is 256
    Description: input the tensors of raw feature
    Expectation: cost_time <= predict_time, confidence >= predict_confidence.
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        max_device_memory="31GB")
    fold_infer(256, 94, True)
