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
"""ESM-IF1"""
import os

import pytest
from mindsponge import PipeLine
from mindsponge.pipeline.pipeline import download_config
from mindsponge.common.config_load import load_config


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_esmif1():
    """
    Feature: test esmif1
    Description: None
    Expectation:
    """
    cmd = "wget https://download.mindspore.cn/mindscience/mindsponge/esm/examples/6t36.pdb"
    os.system(cmd)

    pipe = PipeLine(name="ESM_IF1")
    download_config(pipe.config["sampling"], pipe.config_path + "sampling.yaml")
    conf = load_config(pipe.config_path + "sampling.yaml")
    pipe.initialize(conf=conf)
    pipe.model.from_pretrained(ckpt_path="/home/workspace/mindspore_ckpt/ckpt/esm_if1.ckpt")
    res = pipe.predict(data="./6t36.pdb")
    print(res)
