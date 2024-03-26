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
"""ufold"""
import os

import pytest
from mindsponge import PipeLine
from mindsponge.pipeline.pipeline import download_config
from mindsponge.common.config_load import load_config


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ufold():
    """
    Feature: test ufold_eval
    Description: None
    Expectation: assert result.shape == (1, 352, 352)
    """
    cmd = "wget https://download.mindspore.cn/mindscience/mindsponge/ufold/examples/Acid.caps._TRW-240015_1-352.ct"
    os.system(cmd)
    pipe = PipeLine(name="UFold")
    download_config(pipe.config["ufold_config"], pipe.config_path + "ufold_config.yaml")
    conf = load_config(pipe.config_path + "ufold_config.yaml")
    conf.is_training = False
    conf.test_ckpt = 'All'
    pipe.initialize(conf=conf)
    pipe.model.from_pretrained()
    data = "./Acid.caps._TRW-240015_1-352.ct"
    result = pipe.predict(data)
    print(result)
    assert result[0].shape == (1, 352, 352)
