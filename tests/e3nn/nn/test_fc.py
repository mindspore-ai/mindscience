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
"""Test cases for FullyConnectedNet"""
import pytest
import numpy as np
from mindspore import Tensor, ops
from mindscience.e3nn.nn.fc import FullyConnectedNet


class TestFullyConnectedNet:
    """Test cases for FullyConnectedNet"""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_fc_basic_creation(self):
        """Test basic creation and parameter initialization"""
        h_list = [4, 10, 6]
        fc = FullyConnectedNet(h_list)

        assert fc.h_list == h_list
        assert len(fc.layer_list) == 2
        assert fc.layer_list[0].h_in == 4 and fc.layer_list[0].h_out == 10
        assert fc.layer_list[1].h_in == 10 and fc.layer_list[1].h_out == 6
        assert fc.weight_numel == 4*10 + 10*6

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_fc_forward_computation(self):
        """Test forward propagation computation correctness"""
        h_list = [3, 4, 2]
        fc = FullyConnectedNet(h_list, act=None, out_act=False)

        x = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))

        # Set fixed weights for verification
        fc.layer_list[0].weight.set_data(Tensor(np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]
        ], dtype=np.float32)))

        fc.layer_list[1].weight.set_data(Tensor(np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8]
        ], dtype=np.float32)))

        output = fc(x)

        # Manual calculation verification
        w1_norm = fc.layer_list[0].weight.asnumpy() / np.sqrt(3)
        hidden = np.dot(x.asnumpy(), w1_norm)
        w2_norm = fc.layer_list[1].weight.asnumpy() / np.sqrt(4)
        expected_output = np.dot(hidden, w2_norm)

        assert output.shape == (2,)
        assert np.allclose(output.asnumpy(), expected_output, atol=1e-6)

    def test_fc_activation_function(self):
        """Test activation function"""
        h_list = [2, 3, 2]
        fc_with_act = FullyConnectedNet(h_list, act=ops.tanh, out_act=True)
        fc_without_act = FullyConnectedNet(h_list, act=ops.tanh, out_act=False)

        x = Tensor(np.array([1.0, -1.0], dtype=np.float32))
        output_with_act = fc_with_act(x)
        output_without_act = fc_without_act(x)

        assert output_with_act.shape == (2,)
        assert output_without_act.shape == (2,)
        assert not np.allclose(output_with_act.asnumpy(), output_without_act.asnumpy())

    def test_fc_error_handling(self):
        """Test error handling"""
        # Test invalid h_list
        with pytest.raises(TypeError):
            FullyConnectedNet([3.5, 4, 2])

        # Test minimum valid case
        fc_minimal = FullyConnectedNet([2, 1])
        assert len(fc_minimal.layer_list) == 1


if __name__ == "__main__":
    pytest.main([__file__])
