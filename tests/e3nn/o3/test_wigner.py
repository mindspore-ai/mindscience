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
"""Test cases for o3.wigner module."""

import pytest
import numpy as np
from mindspore import float32, float64, complex64, complex128

from mindscience.e3nn.o3.wigner import (
    change_basis_real_to_complex,
    su2_generators,
    so3_generators,
    wigner_D,
    wigner_3j
)

class TestWigner:
    """Test wigner module functions."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_change_basis_real_to_complex(self):
        """Test change_basis_real_to_complex function."""
        # Test basic functionality
        result = change_basis_real_to_complex(1)
        assert result.shape == (3, 3)
        assert result.dtype == complex64

        # Test dtype conversion
        result = change_basis_real_to_complex(1, dtype=float64)
        assert result.dtype == complex128

        # Test unitarity property
        q_matrix = change_basis_real_to_complex(1)
        q_np = q_matrix.asnumpy()
        identity = np.eye(3)
        np.testing.assert_allclose(q_np @ q_np.conj().T, identity, atol=1e-6)

    def test_su2_generators(self):
        """Test su2_generators function."""
        # Test basic functionality
        result = su2_generators(1)
        assert result.shape == (3, 3, 3)
        assert result.dtype == complex64

        # Test dtype
        result = su2_generators(1, dtype=complex128)
        assert result.dtype == complex128

        # Test invalid input
        with pytest.raises(TypeError):
            su2_generators(1.5)

    def test_so3_generators(self):
        """Test so3_generators function."""
        # Test basic functionality
        result = so3_generators(1)
        assert result.shape == (3, 3, 3)
        assert result.dtype == float32

        # Test dtype
        result = so3_generators(1, dtype=float64)
        assert result.dtype == float64

        # Test invalid input
        with pytest.raises(TypeError):
            so3_generators(1.5)

    def test_wigner_d(self):
        """Test wigner_D function."""
        # Test identity rotation
        result = wigner_D(1, 0, 0, 0)
        assert result.shape == (3, 3)
        expected = np.eye(3)
        np.testing.assert_allclose(result.asnumpy(), expected, atol=1e-6)

        # Test orthogonality property
        d_matrix = wigner_D(1, 0.5, 0.3, 0.7)
        identity = np.eye(3)
        np.testing.assert_allclose((d_matrix @ d_matrix.T).asnumpy(), identity, atol=1e-5)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_wigner_3j(self):
        """Test wigner_3j function."""
        # Test basic functionality
        result = wigner_3j(1, 1, 1)
        assert result.shape == (3, 3, 3)
        assert result.dtype == float32

        # Test dtype
        result = wigner_3j(1, 1, 0, dtype=float64)
        assert result.dtype == float64

        # Test normalization property
        coeffs = wigner_3j(1, 1, 1)
        norm_squared = np.sum(coeffs.asnumpy() ** 2)
        np.testing.assert_allclose(norm_squared, 1.0, atol=1e-6)

        # Test invalid combinations
        with pytest.raises(ValueError):
            wigner_3j(1, 1, 3)

        with pytest.raises(TypeError):
            wigner_3j(1.5, 1, 1)
