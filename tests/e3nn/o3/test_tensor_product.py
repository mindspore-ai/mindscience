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
"""Test cases for tensor_product module."""

import pytest
import numpy as np
from mindspore import Tensor, float32

from mindscience.e3nn.o3.tensor_product import TensorProduct
from mindscience.e3nn.o3.sub import (
    FullTensorProduct, FullyConnectedTensorProduct,
    ElementwiseTensorProduct, TensorSquare, Linear
)
from mindscience.e3nn.o3.irreps import Irreps


class TestTensorProduct:
    """Test class for TensorProduct and related classes."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_tensor_product_basic(self):
        """Test basic TensorProduct functionality."""
        # Test standard tensor product
        tp = TensorProduct('2x1o+1x0e', '1x1o+1x0e')
        assert tp.irreps_in1.dim == 7  # 2*3 + 1*1 = 7
        assert tp.irreps_in2.dim == 4  # 1*3 + 1*1 = 4

        # Test with input tensors
        x1 = Tensor(np.random.randn(2, tp.irreps_in1.dim), dtype=float32)
        x2 = Tensor(np.random.randn(2, tp.irreps_in2.dim), dtype=float32)
        output = tp(x1, x2)
        assert output.shape == (2, tp.irreps_out.dim)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_full_tensor_product(self):
        """Test FullTensorProduct functionality."""
        # Test full tensor product
        ftp = FullTensorProduct('1x1o+1x0e', '1x1o+1x0e')
        x1 = Tensor(np.random.randn(2, ftp.irreps_in1.dim), dtype=float32)
        x2 = Tensor(np.random.randn(2, ftp.irreps_in2.dim), dtype=float32)
        output = ftp(x1, x2)
        assert output.shape == (2, ftp.irreps_out.dim)

    def test_fully_connected_tensor_product(self):
        """Test FullyConnectedTensorProduct functionality."""
        # Test fully connected tensor product
        fctp = FullyConnectedTensorProduct('1x1o', '1x1o', '1x2e+1x0e')
        x1 = Tensor(np.random.randn(2, fctp.irreps_in1.dim), dtype=float32)
        x2 = Tensor(np.random.randn(2, fctp.irreps_in2.dim), dtype=float32)
        output = fctp(x1, x2)
        assert output.shape == (2, fctp.irreps_out.dim)
        assert fctp.weight_numel > 0  # Should have learnable weights

    def test_elementwise_tensor_product(self):
        """Test ElementwiseTensorProduct functionality."""
        # Test elementwise tensor product
        etp = ElementwiseTensorProduct('2x1o+1x0e', '2x1o+1x0e')
        x1 = Tensor(np.random.randn(2, etp.irreps_in1.dim), dtype=float32)
        x2 = Tensor(np.random.randn(2, etp.irreps_in2.dim), dtype=float32)
        output = etp(x1, x2)
        assert output.shape == (2, etp.irreps_out.dim)

    def test_tensor_square(self):
        """Test TensorSquare functionality."""
        # Test tensor square without output specification
        ts = TensorSquare('1x1o+1x0e')
        x = Tensor(np.random.randn(2, ts.irreps_in1.dim), dtype=float32)
        output = ts(x)
        assert output.shape == (2, ts.irreps_out.dim)

        # Test tensor square with output specification
        ts_out = TensorSquare('1x1o', irreps_out='1x2e+1x0e')
        x = Tensor(np.random.randn(2, ts_out.irreps_in1.dim), dtype=float32)
        output = ts_out(x)
        assert output.shape == (2, ts_out.irreps_out.dim)
        assert ts_out.weight_numel > 0  # Should have learnable weights

    def test_linear_operation(self):
        """Test Linear operation functionality."""
        # Test linear operation
        linear = Linear('1x1o+1x0e', '2x1o+1x0e')
        x = Tensor(np.random.randn(2, linear.irreps_in1.dim), dtype=float32)
        output = linear(x)
        assert output.shape == (2, linear.irreps_out.dim)
        assert linear.weight_numel > 0  # Should have learnable weights

    def test_tensor_product_properties(self):
        """Test tensor product properties and edge cases."""
        # Test properties
        tp = TensorProduct('1x1o', '1x1o', '1x2e+1x0e', instructions='connect')
        assert isinstance(tp.irreps_in1, Irreps)
        assert isinstance(tp.irreps_in2, Irreps)
        assert isinstance(tp.irreps_out, Irreps)
        assert isinstance(tp.instructions, list)
        assert tp.weight_numel >= 0

        # Test string representation
        repr_str = repr(tp)
        assert 'TensorProduct' in repr_str
        assert 'connect' in repr_str

        # Test with single batch
        x1 = Tensor(np.random.randn(tp.irreps_in1.dim), dtype=float32)
        x2 = Tensor(np.random.randn(tp.irreps_in2.dim), dtype=float32)
        output = tp(x1, x2)
        assert output.shape == (tp.irreps_out.dim,)
