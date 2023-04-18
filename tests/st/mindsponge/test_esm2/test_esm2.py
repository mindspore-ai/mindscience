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
"""esm2 script"""
import pytest
import numpy as np
from mindsponge.pipeline import PipeLine


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_esm2():
    """
    Feature: esm2 model test
    Description: input the sequence
    Expectation: expect == result.
    """
    expect = np.array([[4.215e-03, 8.575e-03, 1.584e-04, 9.387e-02, 3.970e-02, 5.447e-03],
                       [8.575e-03, 2.515e-02, 3.482e-02, 6.711e-05, 8.125e-03, 3.000e-02],
                       [1.584e-04, 3.482e-02, 2.495e-01, 1.875e-02, 6.557e-05, 5.420e-02],
                       [9.387e-02, 6.711e-05, 1.875e-02, 1.909e-01, 8.887e-02, 4.470e-05],
                       [3.970e-02, 8.125e-03, 6.557e-05, 8.887e-02, 5.402e-03, 1.141e-02],
                       [5.447e-03, 3.000e-02, 5.420e-02, 4.470e-05, 1.141e-02, 2.641e-02]])
    pipeline = PipeLine('ESM2')
    pipeline.initialize(config_path='./esm2_config.yaml')
    pipeline.model.from_pretrained("/home/workspace/mindspore_ckpt/ckpt/esm2.ckpt")
    data = [("protein3", "K A <mask> I S Q")]
    pipeline.dataset.set_data(data)
    protein_data = pipeline.dataset.protein_data
    kwargs = {"return_contacts": True}
    _, _, _, contacts = pipeline.predict(protein_data, **kwargs)
    contacts = contacts.asnumpy()
    tokens_len = pipeline.dataset.batch_lens[0]
    attention_contacts = contacts[0]
    matrix = attention_contacts[: tokens_len, : tokens_len]
    print(matrix)
    print(matrix - expect)
    assert np.allclose(matrix, expect, atol=1e-4)
