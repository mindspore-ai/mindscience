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
"""Test cases for irreps module"""

import pytest
from mindspore import ops
from mindscience.e3nn.o3 import Irrep, Irreps


class TestIrrep:
    """Test cases for Irrep class"""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_irrep_creation_and_properties(self):
        """Test Irrep creation, properties and basic operations"""
        # Test creation with l and p parameters
        irrep1 = Irrep(0, 1)
        assert irrep1.l == 0
        assert irrep1.p == 1
        assert str(irrep1) == "0e"
        assert irrep1.dim == 1
        assert irrep1.is_scalar() is True

        irrep2 = Irrep(1, -1)
        assert irrep2.l == 1
        assert irrep2.p == -1
        assert str(irrep2) == "1o"
        assert irrep2.dim == 3
        assert irrep2.is_scalar() is False

        # Test creation with string notation
        irrep3 = Irrep("2e")
        assert irrep3.l == 2
        assert irrep3.p == 1
        assert irrep3.dim == 5

        irrep4 = Irrep("3y")
        assert irrep4.l == 3
        assert irrep4.p == -1  # (-1)^3 = -1

        # Test comparison operations
        assert irrep1 == Irrep(0, 1)
        assert irrep1 != irrep2
        assert irrep1 < irrep2  # Compare by l first, then p

    def test_irrep_multiplication_and_arithmetic(self):
        """Test Irrep multiplication and arithmetic operations"""
        irrep1 = Irrep(1, 1)
        irrep2 = Irrep(1, 1)

        # Test tensor product
        products = list(irrep1 * irrep2)
        expected = [Irrep(0, 1), Irrep(1, 1), Irrep(2, 1)]
        assert products == expected

        # Test with different parities
        irrep3 = Irrep(1, -1)
        products2 = list(irrep1 * irrep3)
        expected2 = [Irrep(0, -1), Irrep(1, -1), Irrep(2, -1)]
        assert products2 == expected2

        # Test arithmetic operations
        result = 3 * irrep1
        assert isinstance(result, Irreps)
        assert result.data[0].mul == 3
        assert result.data[0].ir == irrep1

        result_add = irrep1 + irrep3
        assert isinstance(result_add, Irreps)
        assert len(result_add) == 2

    def test_irrep_error_handling_and_wigner(self):
        """Test Irrep error handling and Wigner D matrix"""
        # Test error handling
        with pytest.raises(ValueError):
            Irrep(-1, 1)  # Negative l

        with pytest.raises(ValueError):
            Irrep(1, 2)  # Invalid parity

        with pytest.raises(ValueError):
            Irrep("invalid")

        # Test Wigner D matrix
        irrep = Irrep(1, -1)
        rotation_matrix = ops.eye(3)
        d_matrix = irrep.wigD_from_matrix(rotation_matrix)
        assert d_matrix.shape == (3, 3)

        # Test error for non-tensor input
        with pytest.raises(TypeError):
            irrep.wigD_from_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class TestIrreps:
    """Test cases for Irreps class"""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_irreps_creation_and_basic_operations(self):
        """Test Irreps creation and basic operations"""
        # Test creation from string
        irreps1 = Irreps("1x0e+2x1o")
        assert len(irreps1) == 2
        assert irreps1.data[0].mul == 1
        assert irreps1.data[0].ir == Irrep(0, 1)
        assert str(irreps1) == "1x0e+2x1o"

        # Test creation from list of tuples
        irreps2 = Irreps([(1, (0, 1)), (2, (1, -1))])
        assert irreps1 == irreps2

        # Test creation from single Irrep
        irreps3 = Irreps(Irrep(1, 1))
        assert len(irreps3) == 1

        # Test empty creation
        irreps4 = Irreps()
        assert not irreps4  # Check if empty
        assert irreps4.dim == 0

        # Test single irrep without multiplicity
        irreps_single = Irreps("1o")
        assert irreps_single.data[0].mul == 1
        assert irreps_single.data[0].ir == Irrep(1, -1)

    def test_irreps_properties_and_slicing(self):
        """Test Irreps properties and slicing operations"""
        irreps = Irreps("2x0e+3x1o+1x2e")

        # Test dimension
        expected_dim = 2 * 1 + 3 * 3 + 1 * 5  # 2 + 9 + 5 = 16
        assert irreps.dim == expected_dim

        # Test slices
        assert len(irreps.slice) == 3
        assert irreps.slice[0] == slice(0, 2)
        assert irreps.slice[1] == slice(2, 11)
        assert irreps.slice[2] == slice(11, 16)

        # Test lmax and num_irreps properties
        assert irreps.lmax == 2
        assert irreps.num_irreps == 6  # 2 + 3 + 1

        # Test contains operation
        assert Irrep(0, 1) in irreps
        assert Irrep(3, 1) not in irreps

    def test_irreps_arithmetic_and_operations(self):
        """Test Irreps arithmetic operations and advanced features"""
        irreps1 = Irreps("1x0e+1x1o")
        irreps2 = Irreps("2x0e+1x2e")

        # Test addition
        result_add = irreps1 + irreps2
        assert len(result_add) == 4

        # Test multiplication with integer
        result_mul = irreps1 * 2
        expected_mul = Irreps("2x0e+2x1o")
        assert result_mul == expected_mul

        # Test comparison operations
        assert irreps1 == Irreps("1x0e+1x1o")
        assert irreps1 != irreps2

        # Test iteration
        for i, (mul, ir) in enumerate(irreps1):
            if i == 0:
                assert mul == 1 and ir == Irrep(0, 1)
            elif i == 1:
                assert mul == 1 and ir == Irrep(1, -1)

    def test_irreps_error_handling_and_edge_cases(self):
        """Test Irreps error handling and edge cases"""
        # Test invalid string format
        with pytest.raises(ValueError):
            Irreps("invalid_format")

        # Test negative multiplicity
        with pytest.raises(ValueError):
            Irreps([(-1, (0, 1))])

        # Test invalid multiplicity type
        with pytest.raises(ValueError):
            Irreps([(1.5, (0, 1))])

        # Test empty Irreps lmax property
        irreps_empty = Irreps("")
        with pytest.raises(ValueError):
            _ = irreps_empty.lmax

        # Test zero multiplicity
        zero_irreps = Irreps("0x1o+2x0e")
        assert len(zero_irreps) == 2
        assert zero_irreps.data[0].mul == 0

        # Test large irreps
        large_irreps = Irreps("100x0e+50x1e")
        assert large_irreps.dim == 100 * 1 + 50 * 3
        assert len(large_irreps) == 2


class TestMulIr:
    """Test cases for _MulIr class"""

    def test_mulir_comprehensive(self):
        """Test _MulIr creation, properties and operations"""
        from mindscience.e3nn.o3.irreps import _MulIr

        # Test creation and properties
        irrep = Irrep(1, 1)
        mulir = _MulIr(3, irrep)
        assert mulir.mul == 3
        assert mulir.ir == irrep
        assert mulir.dim == 3 * 3  # mul * irrep.dim
        assert str(mulir) == "3x1e"

        # Test iteration/deconstruction
        mul, ir = mulir
        assert mul == 3
        assert ir == irrep

        # Test comparison operations
        mulir2 = _MulIr(3, irrep)
        mulir3 = _MulIr(2, irrep)
        mulir4 = _MulIr(3, Irrep(2, 1))

        assert mulir == mulir2
        assert mulir != mulir3
        assert mulir < mulir4  # Compare by irrep first

        # Test error handling
        with pytest.raises(TypeError):
            _MulIr(1.5, irrep)  # mul should be int

        with pytest.raises(TypeError):
            _MulIr(2, "1e")  # ir should be Irrep instance
