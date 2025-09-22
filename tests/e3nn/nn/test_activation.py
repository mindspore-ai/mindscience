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
"""Test cases for e3nn.nn.activation module - Core functionality only"""

import pytest
import numpy as np
from mindspore import Tensor, ops, float32
from mindscience.e3nn.nn.activation import Activation, _Normalize, _moment, _parity_function
from mindscience.e3nn.o3 import Irreps


class TestActivation:
    """Core tests for Activation class"""

    def test_activation_basic_creation(self):
        """Test basic Activation creation and forward pass"""
        act = Activation('2x0e+1x0o', [ops.tanh, ops.abs])

        x = Tensor(np.random.randn(3, 3), dtype=float32)
        output = act(x)

        assert output.shape == (3, 3)
        assert act.irreps_in == Irreps('2x0e+1x0o')
        assert not np.any(np.isnan(output.asnumpy()))

    def test_activation_parity_change(self):
        """Test activation function changes parity correctly"""
        # abs function should change odd to even
        act = Activation('2x0o', [ops.abs])

        x = Tensor(np.random.randn(2, 2), dtype=float32)
        output = act(x)

        assert act.irreps_out == Irreps('2x0e')  # odd -> even
        assert output.shape == (2, 2)

    def test_activation_invalid_non_scalar(self):
        """Test activation with non-scalar irrep raises error"""
        with pytest.raises(ValueError, match="non-scalar input"):
            Activation('1x1e', [ops.tanh])


class TestNormalize:
    """Core tests for _Normalize class"""

    def test_normalize_basic(self):
        """Test _Normalize normalizes activation function"""
        norm_tanh = _Normalize(ops.tanh)

        x = Tensor(np.random.randn(100), dtype=float32)
        output = norm_tanh(x)

        assert output.shape == x.shape
        assert hasattr(norm_tanh, 'factor')

    def test_normalize_scaling_function(self):
        """Test _Normalize correctly handles scaling functions"""
        def scale_func(x):
            return x * 2.0  # This will have second moment = 4.0

        norm_func = _Normalize(scale_func)

        # Verify factor is approximately correct (should be around 1/sqrt(4) = 0.5)
        expected_factor = 1.0 / np.sqrt(4.0)
        assert abs(float(norm_func.factor) - expected_factor) < 5e-3

        # Test normalization effect
        x = Tensor(np.ones(5), dtype=float32)
        output = norm_func(x)
        expected_output = scale_func(x) * norm_func.factor
        assert np.allclose(output.asnumpy(), expected_output.asnumpy(), atol=1e-4)


class TestUtilityFunctions:
    """Core tests for utility functions"""

    def test_moment_calculation(self):
        """Test _moment function calculates moments correctly"""
        moment = _moment(ops.tanh, 2)

        assert isinstance(moment, Tensor)
        assert moment.shape == ()  # scalar
        assert moment.asnumpy() > 0

    def test_parity_function_detection(self):
        """Test _parity_function detects function parity"""
        # Test even function
        parity_even = _parity_function(lambda x: x**2)
        assert parity_even == 1  # even function

        # Test odd function
        parity_odd = _parity_function(lambda x: x)
        assert parity_odd == -1  # odd function


if __name__ == "__main__":
    pytest.main([__file__])
