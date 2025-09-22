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
"""
Test cases for e3nn.utils module.

This module contains comprehensive test cases for all utility functions
in the e3nn.utils package, including tensor operations, linear algebra,
tensor contractions, radius computations, and initialization utilities.
"""

import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal

from mindscience.e3nn.utils.func import broadcast_args, _ndexpm, narrow
from mindscience.e3nn.utils.linalg import _direct_sum
from mindscience.e3nn.utils.ncon import Ncon
from mindscience.e3nn.utils.radius import radius, radius_graph, radius_full
from mindscience.e3nn.utils.initializer import Uniform, renormal_initializer
from mindscience.e3nn.utils.perm import _from_int, _to_int, _inverse, _compose, _group, _germinate

class TestFuncModule:
    """Test cases for func.py module."""

    def test_broadcast_and_operations(self):
        """Test broadcasting, matrix exponential, and tensor slicing."""
        # Test broadcasting
        a = Tensor([1.0, 2.0])
        b = Tensor([[3.0], [4.0]])
        result = broadcast_args(a, b)
        assert len(result) == 2 and result[0].shape == (2, 2)

        # Test matrix exponential
        mat = Tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=ms.float32)
        exp_result = _ndexpm(mat)
        assert exp_result.shape == (2, 2)

        # Test tensor slicing
        x = Tensor(np.arange(24).reshape(2, 3, 4), dtype=ms.float32)
        sliced = narrow(x, axis=0, start=0, length=1)
        assert sliced.shape == (1, 3, 4)

class TestLinalgModule:
    """Test cases for linalg.py module."""

    def test_direct_sum(self):
        """Test direct sum of matrices."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0]])
        result = _direct_sum(a, b)
        assert result.shape == (3, 3)

        # Test with batch dimensions
        batch_a = Tensor(np.random.randn(2, 3, 3).astype(np.float32))
        batch_b = Tensor(np.random.randn(2, 2, 2).astype(np.float32))
        batch_result = _direct_sum(batch_a, batch_b)
        assert batch_result.shape == (2, 5, 5)

class TestNconModule:
    """Test cases for ncon.py module."""

    def test_ncon_operations(self):
        """Test various Ncon tensor contraction operations."""
        # Test trace
        ncon_trace = Ncon([[1, 1]])
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        trace_result = ncon_trace([a])
        assert np.isclose(trace_result.asnumpy(), 5.0)

        # Test outer product
        ncon_outer = Ncon([[-1], [-2]])
        b = Tensor([1.0, 2.0])
        c = Tensor([3.0, 4.0, 5.0])
        outer_result = ncon_outer([b, c])
        assert outer_result.shape == (2, 3)

        # Test batch matrix multiplication
        ncon_matmul = Ncon([[-1, -2, 1], [-1, 1, -3]])
        d = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
        e = Tensor(np.random.randn(2, 4, 5).astype(np.float32))
        matmul_result = ncon_matmul([d, e])
        assert matmul_result.shape == (2, 3, 5)

class TestRadiusModule:
    """Test cases for radius.py module."""

    def test_radius_functions(self):
        """Test radius computation functions."""
        np.random.seed(42)
        x = np.random.random((8, 3)).astype(np.float32)
        y = np.random.random((5, 3)).astype(np.float32)

        # Test basic radius
        edge_index, batch_x, _ = radius(x, y, 0.5, max_num_neighbors=10)
        assert edge_index.shape[0] == 2 and len(batch_x) == len(x)

        # Test radius_graph
        edge_index, batch = radius_graph(x, 0.8, loop=False)
        assert edge_index.shape[0] == 2 and len(batch) == len(x)

        # Test radius_full
        x_batch = np.ones((2, 3, 3), dtype=np.float32)
        edge_index_full, batch_x_full, _ = radius_full(x_batch, x_batch)
        assert edge_index_full.shape[0] == 2 and len(batch_x_full) == 6

class TestInitializerModule:
    """Test cases for initializer.py module."""

    def test_initializers(self):
        """Test custom initializers."""
        # Test Uniform initializer
        from mindspore.common.initializer import initializer
        uniform_init = Uniform(scale=2.0)
        tensor = initializer(uniform_init, [3, 4], ms.float32)
        values = tensor.asnumpy()
        assert np.all(values >= 0.0) and np.all(values <= 2.0)

        # Test renormal_initializer
        init1 = renormal_initializer('uniform')
        assert isinstance(init1, Uniform)

        init2 = renormal_initializer('truncatedNormal')
        assert isinstance(init2, TruncatedNormal)

        # Test invalid input
        with pytest.raises(ValueError):
            renormal_initializer('invalid_method')

class TestPermModule:
    """Test cases for perm.py module."""

    def test_permutation_operations(self):
        """Test permutation conversion and operations."""
        # Test conversion functions
        n = 3
        for i in range(6):  # 3! = 6
            perm = _from_int(i, n)
            assert len(perm) == n and _to_int(perm) == i

        # Test permutation operations
        perm1 = (0, 2, 1)
        inv_perm1 = _inverse(perm1)
        composed = _compose(perm1, inv_perm1)
        assert composed == (0, 1, 2)  # identity

        # Test group operations
        group3 = _group(3)
        assert len(group3) == 6  # 3! = 6

        subset = {(0, 1, 2), (1, 0, 2)}
        closure = _germinate(subset)
        assert len(closure) >= len(subset)

class TestInputValidation:
    """Test input validation and error handling."""

    def test_error_handling(self):
        """Test various error conditions."""
        # Test radius with mismatched dimensions
        x = np.random.random((5, 3))
        y = np.random.random((5, 4))  # Different last dimension
        with pytest.raises(ValueError):
            radius(x, y, 1.0)

        # Test radius_graph with invalid flow
        with pytest.raises(ValueError):
            radius_graph(x, 1.0, flow='invalid_flow')

        # Test _ndexpm with invalid input
        invalid_mat = Tensor([1.0])  # 1D tensor
        with pytest.raises(ValueError):
            _ndexpm(invalid_mat)

if __name__ == "__main__":
    pytest.main([__file__])
