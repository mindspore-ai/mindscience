"""Test Gate module"""
import pytest
from mindspore import Tensor, ops, float32
import numpy as np
from mindscience.e3nn.nn import Gate


class TestGate:
    """Test cases for Gate module"""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_gate_creation(self):
        """Test Gate creation and basic properties"""
        gate = Gate('2x0e', [ops.tanh], '1x0e', [ops.sigmoid], '1x1o')
        assert isinstance(gate, Gate)
        assert gate.irreps_in.dim > 0
        assert gate.irreps_out.dim > 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_gate_forward(self):
        """Test forward propagation"""
        gate = Gate('1x0e', [ops.tanh], '2x0e', [ops.sigmoid, ops.abs], '2x1o')
        x = Tensor(np.random.randn(3, gate.irreps_in.dim), dtype=float32)
        output = gate(x)

        assert output.shape == (3, gate.irreps_out.dim)
        assert not np.isnan(output.asnumpy()).any()

    def test_gate_activations(self):
        """Test different activation functions"""
        gate1 = Gate('1x0e', [ops.tanh], '1x0e', [ops.sigmoid], '1x1o')
        gate2 = Gate('1x0e', [ops.relu], '1x0e', [ops.abs], '1x1o')

        x = Tensor(np.random.randn(2, gate1.irreps_in.dim), dtype=float32)
        output1, output2 = gate1(x), gate2(x)

        assert output1.shape == output2.shape
        assert not np.allclose(output1.asnumpy(), output2.asnumpy(), atol=1e-6)

    def test_gate_errors(self):
        """Test error handling"""
        with pytest.raises(ValueError, match="Scalars must be scalars"):
            Gate('1x1o', [ops.tanh], '1x0e', [ops.sigmoid], '1x1o')

        with pytest.raises(ValueError, match="Gate scalars must be scalars"):
            Gate('1x0e', [ops.tanh], '1x1o', [ops.sigmoid], '1x1o')

        with pytest.raises(ValueError, match="different number"):
            Gate('1x0e', [ops.tanh], '2x0e', [ops.sigmoid, ops.abs], '1x1o')
