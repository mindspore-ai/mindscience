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
"""Test cases for rotation module."""

import math
import pytest
import numpy as np
from mindspore import Tensor, float32

from mindscience.e3nn.o3.rotation import (
    identity_angles, rand_angles, compose_angles,
    matrix_x, matrix_y, matrix_z,
    angles_to_matrix, matrix_to_angles,
    angles_to_xyz, xyz_to_angles
)


class TestRotation:
    """Test class for rotation functions."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_identity_angles(self):
        """Test identity_angles function comprehensively."""
        # Test basic functionality and shapes
        alpha, beta, gamma = identity_angles(2, 3)
        assert alpha.shape == (2, 3)
        assert beta.shape == (2, 3)
        assert gamma.shape == (2, 3)
        assert np.allclose(alpha.asnumpy(), 0.0)
        assert np.allclose(beta.asnumpy(), 0.0)
        assert np.allclose(gamma.asnumpy(), 0.0)

        # Test dtype
        alpha, beta, gamma = identity_angles(2, dtype=float32)
        assert alpha.dtype == float32

        # Test error handling
        with pytest.raises(TypeError):
            identity_angles(1.5)  # Should be int

    def test_rand_angles(self):
        """Test rand_angles function comprehensively."""
        # Test shapes and angle ranges
        alpha, beta, gamma = rand_angles(2, 3)
        assert alpha.shape == (2, 3)
        assert beta.shape == (2, 3)
        assert gamma.shape == (2, 3)
        assert np.all(alpha.asnumpy() >= 0) and np.all(alpha.asnumpy() <= 2 * math.pi)
        assert np.all(beta.asnumpy() >= 0) and np.all(beta.asnumpy() <= math.pi)
        assert np.all(gamma.asnumpy() >= 0) and np.all(gamma.asnumpy() <= 2 * math.pi)

        # Test error handling
        with pytest.raises(TypeError):
            rand_angles(1.5)  # Should be int

    def test_rotation_matrices(self):
        """Test rotation matrix functions (matrix_x, matrix_y, matrix_z)."""
        # Test identity matrices with zero angle
        for matrix_func in [matrix_x, matrix_y, matrix_z]:
            mat = matrix_func(0.0)
            assert np.allclose(mat.asnumpy(), np.eye(3), atol=1e-6)

        # Test specific rotations
        mat_x = matrix_x(math.pi / 2)
        expected_x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        assert np.allclose(mat_x.asnumpy(), expected_x, atol=1e-6)

        mat_y = matrix_y(math.pi / 2)
        expected_y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        assert np.allclose(mat_y.asnumpy(), expected_y, atol=1e-6)

        mat_z = matrix_z(math.pi / 2)
        expected_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(mat_z.asnumpy(), expected_z, atol=1e-6)

        # Test batch operations
        angles = Tensor([0.1, 0.2, 0.3])
        mat = matrix_x(angles)
        assert mat.shape == (3, 3, 3)

    def test_rotation_matrices_orthogonal(self):
        """Test that rotation matrices are orthogonal."""
        angle = 0.5
        for matrix_func in [matrix_x, matrix_y, matrix_z]:
            mat = matrix_func(angle)
            # Check orthogonality: R @ R.T = I
            identity = np.matmul(mat.asnumpy(), mat.asnumpy().T)
            assert np.allclose(identity, np.eye(3), atol=1e-6)
            # Check determinant = 1
            assert np.allclose(np.linalg.det(mat.asnumpy()), 1.0, atol=1e-6)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_angle_matrix_conversion(self):
        """Test angles_to_matrix and matrix_to_angles functions."""
        # Test identity conversion
        mat = angles_to_matrix(0.0, 0.0, 0.0)
        assert np.allclose(mat.asnumpy(), np.eye(3), atol=1e-6)

        # Test roundtrip conversion
        alpha_orig = Tensor([0.1, 0.2, 0.3])
        beta_orig = Tensor([0.4, 0.5, 0.6])
        gamma_orig = Tensor([0.7, 0.8, 0.9])

        mat = angles_to_matrix(alpha_orig, beta_orig, gamma_orig)
        assert mat.shape == (3, 3, 3)
        alpha_new, beta_new, gamma_new = matrix_to_angles(mat)
        assert np.allclose(alpha_orig.asnumpy(), alpha_new.asnumpy(), atol=1e-5)
        assert np.allclose(beta_orig.asnumpy(), beta_new.asnumpy(), atol=1e-5)
        assert np.allclose(gamma_orig.asnumpy(), gamma_new.asnumpy(), atol=1e-5)

    def test_angles_matrix_roundtrip(self):
        """Test roundtrip conversion between angles and matrix."""
        # Test multiple angle sets
        test_angles = [
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6),
            (1.0, 1.5, 2.0),
            (math.pi/4, math.pi/3, math.pi/6)
        ]

        for alpha, beta, gamma in test_angles:
            # Convert angles to matrix and back
            mat = angles_to_matrix(alpha, beta, gamma)
            alpha_rec, beta_rec, gamma_rec = matrix_to_angles(mat)

            # Check if we get back the same angles (within tolerance)
            # Note: Euler angles may have multiple representations
            mat_rec = angles_to_matrix(alpha_rec, beta_rec, gamma_rec)
            assert np.allclose(mat.asnumpy(), mat_rec.asnumpy(), atol=1e-5)

    def test_matrix_to_angles_error(self):
        """Test matrix_to_angles error handling."""
        # Test with non-rotation matrix (determinant != 1)
        invalid_matrix = Tensor(np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
        with pytest.raises(ValueError):
            matrix_to_angles(invalid_matrix)

    def test_angle_operations(self):
        """Test compose_angles, angles_to_xyz, and xyz_to_angles functions."""
        # Test compose_angles with identity
        alpha_comp, beta_comp, gamma_comp = compose_angles(0.0, 0.0, 0.0, 0.1, 0.2, 0.3)
        assert np.allclose(alpha_comp.asnumpy(), 0.1, atol=1e-6)
        assert np.allclose(beta_comp.asnumpy(), 0.2, atol=1e-6)
        assert np.allclose(gamma_comp.asnumpy(), 0.3, atol=1e-6)

        # Test angles_to_xyz and xyz_to_angles roundtrip
        xyz = angles_to_xyz(0.0, 0.0)
        assert np.allclose(xyz.asnumpy(), [0.0, 1.0, 0.0], atol=1e-6)
        alpha, beta = xyz_to_angles(xyz)
        assert np.allclose(alpha.asnumpy(), 0.0, atol=1e-6)
        assert np.allclose(beta.asnumpy(), 0.0, atol=1e-6)

    def test_batch_and_edge_cases(self):
        """Test batch operations and edge cases."""
        # Test batch operations
        alphas = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]]).astype(np.float32))
        betas = Tensor(np.array([[0.5, 0.6], [0.7, 0.8]]).astype(np.float32))
        gammas = Tensor(np.array([[0.9, 1.0], [1.1, 1.2]]).astype(np.float32))
        matrices = angles_to_matrix(alphas, betas, gammas)
        assert matrices.shape == (2, 2, 3, 3)
        # Test edge case: small angles
        mat = angles_to_matrix(1e-8, 1e-8, 1e-8)
        assert np.allclose(mat.asnumpy(), np.eye(3), atol=1e-6)

        # Test edge case: pi angles (should still be valid rotation matrix)
        mat = angles_to_matrix(math.pi, math.pi, math.pi)
        identity = np.matmul(mat.asnumpy(), mat.asnumpy().T)
        assert np.allclose(identity, np.eye(3), atol=1e-5)
        assert np.allclose(np.linalg.det(mat.asnumpy()), 1.0, atol=1e-5)
