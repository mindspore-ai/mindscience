"""Test cases for NormActivation module"""
import pytest
from mindspore import Tensor, ops, float32
import numpy as np
from mindscience.e3nn.nn import NormActivation


class TestNormActivation:
    """Test cases for NormActivation class"""

    def test_creation_and_forward(self):
        """Test NormActivation creation and forward pass"""
        normact = NormActivation('2x1e', ops.sigmoid)
        assert normact.irreps_in.dim > 0
        assert normact.irreps_out.dim == normact.irreps_in.dim
        assert normact.normalize is True
        assert normact.epsilon == 1e-8

        x = Tensor(np.random.randn(3, normact.irreps_in.dim), dtype=float32)
        output = normact(x)
        assert output.shape == x.shape
        assert not np.isnan(output.asnumpy()).any()

    def test_normalize_and_epsilon(self):
        """Test normalize parameter and epsilon configuration"""
        normact_norm = NormActivation('1x1o', ops.sigmoid, normalize=True)
        normact_no_norm = NormActivation('1x1o', ops.sigmoid, normalize=False)
        normact_eps = NormActivation('1x1o', ops.sigmoid, epsilon=1e-6)

        assert normact_norm.normalize and normact_norm.epsilon == 1e-8
        assert not normact_no_norm.normalize and normact_no_norm.epsilon is None
        assert normact_eps.epsilon == 1e-6 and normact_eps.epsilon * normact_eps.epsilon == 1e-12

    def test_activations_and_bias(self):
        """Test different activation functions and bias parameter"""
        normact1 = NormActivation('1x1o', ops.sigmoid, bias=True)
        normact2 = NormActivation('1x1o', ops.tanh, bias=False)

        x = Tensor(np.random.randn(2, 3), dtype=float32)
        output1, output2 = normact1(x), normact2(x)

        assert output1.shape == output2.shape
        assert normact1.bias is not None and normact2.bias is None

    def test_errors(self):
        """Test error handling for invalid parameter combinations"""
        with pytest.raises(ValueError, match="epsilon.*normalize = False.*don't make sense"):
            NormActivation('1x1o', ops.sigmoid, normalize=False, epsilon=1e-6)
        with pytest.raises(ValueError, match="epsilon.*invalid.*strictly positive"):
            NormActivation('1x1o', ops.sigmoid, epsilon=-1e-6)
