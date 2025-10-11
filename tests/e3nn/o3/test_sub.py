# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Test cases for o3.sub module.

This module contains comprehensive tests for all classes and functions in the o3.sub module,
including tensor product operations, linear operations, and utility functions.
"""

import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor

from mindscience.e3nn.o3.sub import (
    FullyConnectedTensorProduct,
    FullTensorProduct,
    ElementwiseTensorProduct,
    Linear,
    LinearBias,
    TensorSquare,
    prod,
    _prod,
    _sum_tensors_withbias,
    Instruction
)


class TestTensorProductClasses:
    """Test tensor product classes functionality."""

    def test_tensor_product_operations(self):
        """Test core tensor product operations."""
        # Test FullyConnectedTensorProduct
        tp_fc = FullyConnectedTensorProduct('1x1o', '1x0e', '1x1o')
        x1 = Tensor(np.random.randn(2, 3), ms.float32)
        x2 = Tensor(np.random.randn(2, 1), ms.float32)
        output_fc = tp_fc(x1, x2)
        assert output_fc.shape == (2, 3)

        # Test FullTensorProduct
        tp_full = FullTensorProduct('1x1o', '1x0e')
        output_full = tp_full(x1, x2)
        assert output_full.ndim == 2

        # Test ElementwiseTensorProduct
        tp_elem = ElementwiseTensorProduct('1x1o', '1x1o')
        output_elem = tp_elem(x1, x1)
        assert output_elem.ndim == 2

    def test_linear_operations(self):
        """Test linear operations with and without bias."""
        # Test Linear
        linear = Linear('1x1o+1x0e', '1x1o')
        x = Tensor(np.random.randn(2, 4), ms.float32)
        output = linear(x)
        assert output.shape == (2, 3)

        # Test LinearBias
        linear_bias = LinearBias('1x1o+1x0e', '1x1o+1x0e', has_bias=True)
        output_bias = linear_bias(x)
        assert output_bias.shape == (2, 4)

    def test_tensor_square(self):
        """Test TensorSquare operation."""
        ts = TensorSquare('1x1o', irreps_out='1x0e+1x2e')
        x = Tensor(np.random.randn(2, 3), ms.float32)
        output = ts(x)
        assert output.shape == (2, 6)  # 1x0e+1x2e has dim 6


class TestUtilityFunctions:
    """Test utility functions."""

    def test_prod_functions(self):
        """Test product computation functions."""
        # Test prod function
        assert prod([2, 3, 4]) == 24
        assert prod([]) == 1

        # Test _prod function
        assert _prod((2, 3, 4)) == 24
        assert _prod(()) == 1

    def test_tensor_utilities(self):
        """Test tensor utility functions."""
        # Test _sum_tensors_withbias
        t1 = Tensor(np.array([1, 2, 3]), ms.float32)
        t2 = Tensor(np.array([4, 5, 6]), ms.float32)

        result = _sum_tensors_withbias([t1, t2], (3,), ms.float32)
        expected = np.array([5, 7, 9])
        assert np.allclose(result.asnumpy(), expected)

        # Test Instruction NamedTuple
        instr = Instruction(i_in=0, i_out=1, path_shape=(2, 3), path_weight=1.5)
        assert instr.i_in == 0 and instr.i_out == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid irreps
        with pytest.raises((ValueError, TypeError)):
            FullyConnectedTensorProduct('invalid', '1x0e', '1x0e')

        # Test mismatched dimensions
        tp = FullyConnectedTensorProduct('1x0e', '1x0e', '1x0e')
        x1 = Tensor(np.random.randn(2, 5), ms.float32)  # Wrong dimension
        x2 = Tensor(np.random.randn(2, 1), ms.float32)

        with pytest.raises(ValueError):
            tp(x1, x2)

    def test_scalar_operations(self):
        """Test operations with scalar irreps."""
        linear = Linear('1x0e', '1x0e')
        x = Tensor(np.random.randn(2, 1), ms.float32)
        output = linear(x)
        assert output.shape == (2, 1)
