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
import mindspore as ms
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
    expect = np.array([[4.1792e-03, 8.5416e-03, 1.5924e-04, 9.4268e-02, 3.9924e-02, 5.4555e-03],
                       [8.5416e-03, 2.5114e-02, 3.4650e-02, 6.7362e-05, 8.1558e-03, 3.0048e-02],
                       [1.5924e-04, 3.4650e-02, 2.4789e-01, 1.8577e-02, 6.6454e-05, 5.4532e-02],
                       [9.4268e-02, 6.7362e-05, 1.8577e-02, 1.9135e-01, 8.8563e-02, 4.4795e-05],
                       [3.9924e-02, 8.1558e-03, 6.6454e-05, 8.8563e-02, 5.3860e-03, 1.1304e-02],
                       [5.4555e-03, 3.0048e-02, 5.4532e-02, 4.4795e-05, 1.1304e-02, 2.6510e-02]])
    pipeline = PipeLine('ESM2')
    pipeline.initialize(config_path='./esm2_config.yaml')
    pipeline.model.network.to_float(ms.float32)
    pipeline.model.from_pretrained("/home/workspace/mindspore_ckpt/ckpt/esm2.ckpt")
    data = [("protein3", "K A <mask> I S Q")]
    pipeline.dataset.set_data(data)
    protein_data = pipeline.dataset.protein_data
    _, _, _, contacts = pipeline.predict(protein_data)
    contacts = contacts.asnumpy()
    tokens_len = pipeline.dataset.batch_lens[0]
    attention_contacts = contacts[0]
    matrix = attention_contacts[: tokens_len, : tokens_len]
    print(matrix)
    print(matrix - expect)
    assert np.allclose(matrix, expect, atol=5e-4)
