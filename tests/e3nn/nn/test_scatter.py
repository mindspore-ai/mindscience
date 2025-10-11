# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test scatter"""
import numpy as np
import pytest
from mindspore import Tensor, float32, int32
from mindscience.e3nn.nn import Scatter


class TestScatter:
    """Test Scatter class core functionality"""

    def test_scatter_add(self):
        """Test scatter add operation"""
        scatter = Scatter(mode='add')

        src = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=float32)
        index = Tensor([0, 1], dtype=int32)

        result = scatter(src, index, dim_size=2)
        expected = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=float32)

        assert np.allclose(result.asnumpy(), expected.asnumpy())

    def test_scatter_max(self):
        """Test scatter max operation"""
        scatter = Scatter(mode='max')

        src = Tensor([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]], dtype=float32)
        index = Tensor([0, 1, 0], dtype=int32)

        result = scatter(src, index, dim_size=2)
        expected = Tensor([[2.0, 5.0], [3.0, 2.0]], dtype=float32)

        assert np.allclose(result.asnumpy(), expected.asnumpy())

    def test_scatter_with_out_parameter(self):
        """Test scatter with out parameter for proper initialization"""
        scatter = Scatter(mode='mul')

        src = Tensor([[2.0, 3.0], [4.0, 5.0]], dtype=float32)
        index = Tensor([0, 1], dtype=int32)
        out = Tensor([[1.0, 1.0], [1.0, 1.0]], dtype=float32)

        result = scatter(src, index, out=out)
        expected = Tensor([[2.0, 3.0], [4.0, 5.0]], dtype=float32)

        assert np.allclose(result.asnumpy(), expected.asnumpy())

    def test_scatter_invalid_mode(self):
        """Test scatter with invalid mode"""
        with pytest.raises(ValueError, match="Unexpected scatter mode"):
            Scatter(mode='invalid')
