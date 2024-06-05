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
"""multimer script"""
import pickle
import pytest
from mindsponge import PipeLine
from mindsponge.common.config_load import load_config


@pytest.mark.level0
@pytest.mark.platfrom_arm_ascend910b_training
@pytest.mark.env_onecard
def test_multimer_seq_256():
    """
    Feature: multimer model test, seq length is 256
    Description: input the tensors of raw feature
    Expectation: cost_time <= predict_time, confidence >= predict_confidence.
    """
    pipe = PipeLine(name="Multimer")
    conf = load_config("./predict_256.yaml")
    pipe.initialize(conf=conf)
    pipe.model.from_pretrained(ckpt_path='/home/workspace/mindspore_ckpt/ckpt/Multimer_Model_1.ckpt')
    f = open("/home/workspace/mindspore_dataset/mindsponge_data/6T36/6T36.pkl", "rb")
    raw_feature = pickle.load(f)
    f.close()
    _, _, confidence, _ = pipe.predict(raw_feature)
    assert confidence > 92
    print("confidence:", confidence)
