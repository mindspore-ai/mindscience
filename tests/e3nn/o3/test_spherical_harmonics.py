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
"""Test spherical harmonics module."""

import pytest
import numpy as np
from mindspore import Tensor, float32
from mindscience.e3nn.o3 import spherical_harmonics, SphericalHarmonics


class TestSphericalHarmonicsFunction:
    """Test spherical_harmonics function."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_core_functionality(self):
        """Test core spherical harmonics functionality including degrees and normalization."""
        x = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], float32)

        # Test l=0 (constant function)
        result_l0 = spherical_harmonics(0, x)
        assert result_l0.shape == (2, 1)
        np.testing.assert_allclose(result_l0.asnumpy(), [[0.28209479], [0.28209479]], rtol=1e-5)

        # Test different degrees
        result_l1 = spherical_harmonics(1, x[:1])
        assert result_l1.shape == (1, 3)
        result_l2 = spherical_harmonics(2, x[:1])
        assert result_l2.shape == (1, 5)

        # Test multiple degrees
        result_multi = spherical_harmonics([0, 1, 2], x[:1])
        assert result_multi.shape == (1, 9)  # 1 + 3 + 5

    def test_normalization_and_parameters(self):
        """Test normalization methods and normalize parameter."""
        x = Tensor([[1.0, 0.0, 0.0]], float32)
        x_unnorm = Tensor([[2.0, 0.0, 0.0]], float32)

        # Test different normalization methods
        result_integral = spherical_harmonics(1, x, normalization='integral')
        result_component = spherical_harmonics(1, x, normalization='component')
        result_norm = spherical_harmonics(1, x, normalization='norm')

        # Results should be different for different normalizations
        assert not np.allclose(result_integral.asnumpy(), result_component.asnumpy())
        assert not np.allclose(result_integral.asnumpy(), result_norm.asnumpy())

        # Test normalize parameter
        result_normalized = spherical_harmonics(1, x_unnorm, normalize=True)
        result_unnormalized = spherical_harmonics(1, x_unnorm, normalize=False)
        assert not np.allclose(result_normalized.asnumpy(), result_unnormalized.asnumpy())

    def test_batch_and_shapes(self):
        """Test batch processing and different input shapes."""
        # Multiple vectors
        x = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float32)
        result = spherical_harmonics(2, x)
        assert result.shape == (3, 5)

        # Higher dimensional batch
        x_batch = Tensor(np.random.randn(2, 3, 3).astype(np.float32))
        result_batch = spherical_harmonics(1, x_batch)
        assert result_batch.shape == (2, 3, 3)

        # 3D input
        x_3d = Tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], float32)
        result_3d = spherical_harmonics(1, x_3d)
        assert result_3d.shape == (1, 2, 3)


class TestSphericalHarmonicsClass:
    """Test the SphericalHarmonics class."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_class_initialization_and_forward(self):
        """Test class initialization and forward computation."""
        # Test initialization
        sh = SphericalHarmonics(2, normalize=True)
        # Verify the output dimension instead of accessing protected member
        assert sh.irreps_out.dim == 5
        # Verify the irreps_out contains the expected l=2 representation
        assert str(sh.irreps_out) == "1x2e"

        # Test forward computation
        x = Tensor([[1.0, 0.0, 0.0]], float32)
        result = sh(x)
        assert result.shape == (1, 5)

        # Compare with function version
        result_func = spherical_harmonics(2, x)
        np.testing.assert_allclose(result.asnumpy(), result_func.asnumpy(), rtol=1e-5)

    def test_consistency_and_parity(self):
        """Test normalization consistency and parity."""
        x = Tensor([[1.0, 0.0, 0.0]], float32)

        # Test normalization consistency
        sh_integral = SphericalHarmonics(1, normalize=True, normalization='integral')
        sh_component = SphericalHarmonics(1, normalize=True, normalization='component')
        result_integral = sh_integral(x)
        result_component = sh_component(x)
        assert not np.allclose(result_integral.asnumpy(), result_component.asnumpy())

        # Test parity consistency
        sh = SphericalHarmonics(2, normalize=True)
        x_pos = Tensor([[1.0, 0.0, 0.0]], float32)
        x_neg = Tensor([[-1.0, 0.0, 0.0]], float32)
        result_pos = sh(x_pos)
        result_neg = sh(x_neg)
        # For even l, parity should be preserved
        np.testing.assert_allclose(result_pos.asnumpy(), result_neg.asnumpy(), rtol=1e-5)


class TestMathematicalProperties:
    """Test mathematical properties of spherical harmonics."""

    def test_mathematical_properties(self):
        """Test basic mathematical properties and rotation equivariance."""
        # Test basic properties for l=1
        x = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float32)
        result = spherical_harmonics(1, x)

        # Test that results are finite and have correct shape
        assert result.shape == (3, 3)
        assert np.all(np.isfinite(result.asnumpy()))

        # Test rotation equivariance (simplified)
        x_original = Tensor([[1.0, 0.0, 0.0]], float32)
        x_rotated = Tensor([[0.0, 1.0, 0.0]], float32)  # 90° rotation around z
        sh_original = spherical_harmonics(1, x_original)
        sh_rotated = spherical_harmonics(1, x_rotated)
        # Results should be different for different orientations
        assert sh_original.shape == sh_rotated.shape
        assert not np.allclose(sh_original.asnumpy(), sh_rotated.asnumpy())


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_edge_cases(self):
        """Test zero vectors, high degrees, and error conditions."""
        # Test zero vector
        x_zero = Tensor([[0.0, 0.0, 0.0]], float32)
        result_zero = spherical_harmonics(1, x_zero)
        assert result_zero.shape == (1, 3)

        # Test high degree
        x = Tensor([[1.0, 0.0, 0.0]], float32)
        result_high = spherical_harmonics(5, x)
        assert result_high.shape == (1, 11)  # 2*5+1

        # Test invalid degree (should raise error)
        try:
            spherical_harmonics(-1, x)
            assert False, "Should raise error for negative degree"
        except (ValueError, TypeError):
            pass

        # Test invalid normalization
        try:
            spherical_harmonics(1, x, normalization='invalid')
            assert False, "Should raise error for invalid normalization"
        except (ValueError, TypeError):
            pass
