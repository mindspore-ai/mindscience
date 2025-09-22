"""Test cases for e3nn.o3.norm module - Streamlined core functionality"""
import pytest
import numpy as np
from mindspore import Tensor, float32

from mindscience.e3nn.o3 import Norm, Irreps


class TestNorm:
    """Streamlined tests for Norm class"""

    def test_norm_creation_and_basic_properties(self):
        """Test Norm creation with different irreps and basic properties"""
        # Test basic creation with string irreps
        norm1 = Norm('1x0e')
        assert norm1.irreps_in == Irreps('1x0e')
        assert norm1.irreps_out == Irreps('1x0e')
        assert not norm1.squared

        # Test creation with Irreps object and squared parameter
        irreps_in = Irreps('2x1o + 3x0e')
        norm2 = Norm(irreps_in, squared=True)
        assert norm2.irreps_in == irreps_in.simplify()
        assert norm2.irreps_out == Irreps('2x0e + 3x0e').simplify()
        assert norm2.squared

        # Test string representation
        repr_str = repr(norm2)
        assert 'Norm' in repr_str

    def test_norm_forward_pass_comprehensive(self):
        """Test forward pass with various irrep types and configurations"""
        # Test scalar irrep (0e)
        norm_scalar = Norm('2x0e')
        scalar_input = Tensor([1.0, -2.0], dtype=float32)
        scalar_output = norm_scalar(scalar_input)
        np.testing.assert_allclose(scalar_output.asnumpy(), [1.0, 2.0], rtol=1e-5)

        # Test vector irrep (1o) with batch processing
        norm_vector = Norm('1x1o')
        vector_batch = Tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=float32)
        vector_output = norm_vector(vector_batch)
        expected = np.array([[5.0], [0.0]])
        np.testing.assert_allclose(vector_output.asnumpy(), expected, rtol=1e-5)

        # Test mixed irreps
        norm_mixed = Norm('1x0e + 1x1o')
        mixed_input = Tensor([2.0, 3.0, 4.0, 0.0], dtype=float32)
        mixed_output = norm_mixed(mixed_input)
        expected_mixed = np.array([2.0, 5.0])  # scalar norm + vector norm
        np.testing.assert_allclose(mixed_output.asnumpy(), expected_mixed, rtol=1e-5)

    def test_norm_squared_and_dtype_consistency(self):
        """Test squared parameter and dtype consistency"""
        # Test squared vs regular norm
        norm_regular = Norm('1x1o', squared=False, dtype=float32)
        norm_squared = Norm('1x1o', squared=True, dtype=float32)

        input_vec = Tensor([3.0, 4.0, 0.0], dtype=float32)
        output_regular = norm_regular(input_vec)
        output_squared = norm_squared(input_vec)

        # Verify squared relationship and dtype consistency
        np.testing.assert_allclose(output_regular.asnumpy(), [5.0], rtol=1e-5)
        np.testing.assert_allclose(output_squared.asnumpy(), [25.0], rtol=1e-5)
        assert output_regular.dtype == float32
        assert output_squared.dtype == float32

    def test_norm_mathematical_properties_and_edge_cases(self):
        """Test mathematical properties and edge cases"""
        norm = Norm('1x1o')

        # Test scaling property: ||k*v|| = |k| * ||v||
        vector = Tensor([3.0, 4.0, 0.0], dtype=float32)
        scaled_vector = Tensor([6.0, 8.0, 0.0], dtype=float32)

        norm_original = norm(vector)
        norm_scaled = norm(scaled_vector)
        np.testing.assert_allclose(norm_scaled.asnumpy(), 2.0 * norm_original.asnumpy(), rtol=1e-5)

        # Test with very small values
        small_input = Tensor([1e-10, 1e-10, 1e-10], dtype=float32)
        small_output = norm(small_input)
        expected_small = np.sqrt(3) * 1e-10
        np.testing.assert_allclose(small_output.asnumpy(), [expected_small], rtol=1e-5)

    def test_norm_higher_order_and_mixed_parity(self):
        """Test higher order irreps and mixed parity"""
        # Test l=2 irrep
        norm_l2 = Norm('1x2e')
        l2_input = Tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float32)
        l2_output = norm_l2(l2_input)
        expected_l2 = np.sqrt(5.0)
        np.testing.assert_allclose(l2_output.asnumpy(), [expected_l2], rtol=1e-5)

        # Test mixed parity
        norm_mixed_parity = Norm('1x0e + 1x1o + 1x0o')
        mixed_parity_input = Tensor([2.0, 1.0, 1.0, 1.0, 3.0], dtype=float32)
        mixed_parity_output = norm_mixed_parity(mixed_parity_input)
        expected_mixed_parity = np.array([2.0, np.sqrt(3.0), 3.0])
        np.testing.assert_allclose(mixed_parity_output.asnumpy(), expected_mixed_parity, rtol=1e-5)

    def test_norm_error_handling(self):
        """Test error handling for invalid inputs"""
        norm = Norm('1x1o')

        # Test with wrong input dimension
        with pytest.raises((ValueError, RuntimeError)):
            wrong_dim_input = Tensor([1.0, 2.0], dtype=float32)  # Should be 3D for 1x1o
            norm(wrong_dim_input)
