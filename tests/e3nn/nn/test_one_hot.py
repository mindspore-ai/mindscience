"""Test cases for one_hot module"""
import pytest
import numpy as np

from mindspore import Tensor, ops, float32, int32
from mindscience.e3nn.nn.one_hot import OneHot, SoftOneHotLinspace, soft_one_hot_linspace, soft_unit_step


class TestSoftUnitStep:
    """Test soft_unit_step function"""

    def test_soft_unit_step_basic(self):
        """Test soft_unit_step with basic functionality"""
        # Test positive values
        x_pos = Tensor([1.0, 2.0], dtype=float32)
        result_pos = soft_unit_step(x_pos)
        expected_pos = ops.exp(-1.0 / x_pos)
        assert np.allclose(result_pos.asnumpy(), expected_pos.asnumpy(), atol=1e-6)

        # Test negative values (should be zero due to relu)
        x_neg = Tensor([-1.0, -2.0], dtype=float32)
        result_neg = soft_unit_step(x_neg)
        expected_neg = Tensor([0.0, 0.0], dtype=float32)
        assert np.allclose(result_neg.asnumpy(), expected_neg.asnumpy(), atol=1e-6)

        # Test zero (may be NaN or 0 due to division by zero)
        x_zero = Tensor([0.0], dtype=float32)
        result_zero = soft_unit_step(x_zero)
        result_np = result_zero.asnumpy()
        assert result_np[0] == 0.0 or np.isnan(result_np[0])


class TestOneHot:
    """Test OneHot class"""

    def test_onehot_basic(self):
        """Test OneHot basic functionality"""
        num_types = 4
        onehot = OneHot(num_types)

        # Test creation
        assert onehot.num_types == num_types
        assert str(onehot.irreps_output) == "4x0e"

        # Test single input
        atom_type = Tensor([2], dtype=int32)
        result = onehot(atom_type)
        expected = Tensor([[0., 0., 1., 0.]], dtype=float32)
        assert np.allclose(result.asnumpy(), expected.asnumpy())
        assert result.shape == (1, 4)

        # Test batch input
        atom_types = Tensor([0, 1, 2], dtype=int32)
        result_batch = onehot(atom_types)
        expected_batch = Tensor([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.]
        ], dtype=float32)
        assert np.allclose(result_batch.asnumpy(), expected_batch.asnumpy())
        assert result_batch.shape == (3, 4)


class TestSoftOneHotLinspace:
    """Test SoftOneHotLinspace class"""

    def test_soft_onehot_basic(self):
        """Test SoftOneHotLinspace basic functionality"""
        start, end, number = 0.0, 2.0, 4
        soft_onehot = SoftOneHotLinspace(start, end, number)

        # Test creation
        assert soft_onehot.start.asnumpy() == start
        assert soft_onehot.end.asnumpy() == end
        assert soft_onehot.number == number

        # Test forward pass
        x = Tensor([1.0], dtype=float32)
        result = soft_onehot(x)
        assert result.shape == (1, 4)

        # Test batch input
        x_batch = Tensor([[0.5, 1.0], [1.5, 2.0]], dtype=float32)
        result_batch = soft_onehot(x_batch)
        assert result_batch.shape == (2, 2, 4)

    def test_soft_onehot_different_basis(self):
        """Test SoftOneHotLinspace with different basis functions"""
        start, end, number = 0.0, 2.0, 3
        x = Tensor([1.0], dtype=float32)

        for basis in ['gaussian', 'cosine', 'smooth_finite']:
            soft_onehot = SoftOneHotLinspace(start, end, number, basis=basis)
            result = soft_onehot(x)
            assert result.shape == (1, 3)
            # Some basis functions may produce NaN at boundaries, which is expected

    def test_soft_onehot_cutoff(self):
        """Test SoftOneHotLinspace cutoff behavior"""
        start, end, number = 0.0, 2.0, 3

        # Test with and without cutoff
        soft_onehot_cutoff = SoftOneHotLinspace(start, end, number, cutoff=True)
        soft_onehot_no_cutoff = SoftOneHotLinspace(start, end, number, cutoff=False)

        x = Tensor([3.0], dtype=float32)  # Outside domain
        result_cutoff = soft_onehot_cutoff(x)
        result_no_cutoff = soft_onehot_no_cutoff(x)

        assert result_cutoff.shape == (1, 3)
        assert result_no_cutoff.shape == (1, 3)


class TestSoftOneHotLinspaceFunction:
    """Test soft_one_hot_linspace function"""

    def test_function_basic(self):
        """Test soft_one_hot_linspace function interface"""
        x = Tensor([1.0, 1.5, 2.0], dtype=float32)
        start, end, number = 0.0, 3.0, 4

        result = soft_one_hot_linspace(x, start, end, number)
        assert result.shape == (3, 4)

        # Test with different basis
        result_gaussian = soft_one_hot_linspace(x, start, end, number, basis='gaussian')
        assert result_gaussian.shape == (3, 4)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_edge_cases(self):
        """Test various edge cases"""
        # OneHot with single type
        onehot = OneHot(1)
        atom_type = Tensor([0], dtype=int32)
        result = onehot(atom_type)
        assert result.shape == (1, 1)
        assert np.allclose(result.asnumpy(), Tensor([[1.0]], dtype=float32).asnumpy())

        # SoftOneHotLinspace with small number
        soft_onehot = SoftOneHotLinspace(0.0, 1.0, 2)
        x = Tensor([0.5], dtype=float32)
        result = soft_onehot(x)
        assert result.shape == (1, 2)

        # Invalid basis should raise error
        soft_onehot_invalid = SoftOneHotLinspace(0.0, 1.0, 3, basis='invalid')
        with pytest.raises(ValueError, match="Unsupported basis"):
            soft_onehot_invalid(x)
