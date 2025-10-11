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
"""Test cases for e3nn.nn.batchnorm module - Core functionality"""

import pytest
import numpy as np
from mindspore import Tensor, float32
from mindscience.e3nn.nn.batchnorm import BatchNorm
from mindscience.e3nn.o3 import Irreps


class TestBatchNorm:
    """Core tests for BatchNorm class"""

    def test_batchnorm_basic_creation(self):
        """Test basic BatchNorm creation and forward pass"""
        bn = BatchNorm('2x0e+1x0o')

        x = Tensor(np.random.randn(4, 3), dtype=float32)
        output = bn(x)

        assert output.shape == (4, 3)
        assert bn.irreps == Irreps('2x0e+1x0o')
        assert not np.any(np.isnan(output.asnumpy()))

    def test_batchnorm_normalization_correctness(self):
        """Test that BatchNorm actually normalizes the data correctly"""
        bn = BatchNorm('2x0e', eps=1e-8, affine=False)

        # Create data with known statistics
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
        output = bn(x)

        # Manually compute expected normalized output
        x_np = x.asnumpy()
        x_mean = np.mean(x_np, axis=0)  # [4.0, 5.0]
        x_var = np.var(x_np, axis=0, ddof=0)  # [5.0, 5.0]
        expected_output = (x_np - x_mean) / np.sqrt(x_var + 1e-8)

        # Check that actual output matches manual calculation
        output_np = output.asnumpy()
        assert np.allclose(output_np, expected_output, atol=1e-6), \
            f"Normalization calculation incorrect"

        # Verify normalized output has zero mean and unit variance
        assert abs(np.mean(output_np)) < 1e-6, "Mean should be close to 0"
        assert abs(np.var(output_np, ddof=0) - 1.0) < 1e-5, "Variance should be close to 1"

    def test_batchnorm_affine_parameters(self):
        """Test affine parameters (weight and bias) effect"""
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32))

        # Test with affine=True
        bn_affine = BatchNorm('2x0e', affine=True, eps=1e-8)
        weight = Tensor([2.0, 0.5], dtype=float32)
        bias = Tensor([1.0, -1.0], dtype=float32)
        bn_affine.weight.set_data(weight)
        bn_affine.bias.set_data(bias)

        output_affine = bn_affine(x)

        # Verify computation: output = weight * normalized_input + bias
        x_np = x.asnumpy()
        x_mean = np.mean(x_np, axis=0)
        x_var = np.var(x_np, axis=0, ddof=0)
        x_normalized = (x_np - x_mean) / np.sqrt(x_var + 1e-8)
        expected_output = x_normalized * weight.asnumpy() + bias.asnumpy()

        assert np.allclose(output_affine.asnumpy(), expected_output, atol=1e-5), \
            "Affine transformation calculation incorrect"

        # Test with affine=False
        bn_no_affine = BatchNorm('2x0e', affine=False, eps=1e-8)
        output_no_affine = bn_no_affine(x)

        assert np.allclose(output_no_affine.asnumpy(), x_normalized, atol=1e-5), \
            "Non-affine normalization calculation incorrect"

    def test_batchnorm_training_inference_modes(self):
        """Test difference between training and inference modes"""
        bn = BatchNorm('2x0e', momentum=0.1, instance=False, affine=False)
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32))

        # Training mode - should update running statistics
        bn.training = True
        output_train = bn(x)

        # Verify running statistics update follows momentum formula
        x_np = x.asnumpy()
        batch_mean = np.mean(x_np, axis=0)
        batch_var = np.var(x_np, axis=0, ddof=0)
        expected_running_mean = 0.9 * 0.0 + 0.1 * batch_mean  # initial mean is 0
        expected_running_var = 0.9 * 1.0 + 0.1 * batch_var    # initial var is 1

        assert np.allclose(bn.running_mean.asnumpy(), expected_running_mean, atol=1e-6), \
            "Running mean update calculation incorrect"
        assert np.allclose(bn.running_var.asnumpy(), expected_running_var, atol=1e-6), \
            "Running var update calculation incorrect"

        # Inference mode - should not update running statistics
        running_mean_before = bn.running_mean.asnumpy().copy()
        running_var_before = bn.running_var.asnumpy().copy()

        bn.training = False
        output_inference = bn(x)

        assert np.allclose(bn.running_mean.asnumpy(), running_mean_before), \
            "Running mean should not change in inference mode"
        assert np.allclose(bn.running_var.asnumpy(), running_var_before), \
            "Running var should not change in inference mode"
        assert not np.any(np.isnan(output_train.asnumpy()))
        assert not np.any(np.isnan(output_inference.asnumpy()))

    def test_batchnorm_invalid_parameters(self):
        """Test error handling for invalid parameters"""
        # Test invalid normalization
        with pytest.raises(ValueError, match="Invalid normalization option"):
            bn = BatchNorm('2x0e', normalization='invalid')
            x = Tensor(np.random.randn(4, 2), dtype=float32)
            bn(x)

        # Test invalid reduce
        with pytest.raises(ValueError, match="Invalid reduce option"):
            bn = BatchNorm('2x0e', reduce='invalid')
            x = Tensor(np.random.randn(4, 2), dtype=float32)
            bn(x)


if __name__ == "__main__":
    pytest.main([__file__])
