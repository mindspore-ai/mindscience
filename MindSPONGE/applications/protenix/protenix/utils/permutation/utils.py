# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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
Permutation utils.
"""

import mindspore as ms
from mindspore import mint


class Checker:
    """Checker for permutations."""

    @staticmethod
    def is_permutation(x: ms.Tensor):
        """
        Checks if the input tensor `x` is a permutation of integers from 0 to N-1.

        Args:
            x (ms.Tensor): A 1D tensor of size [N].
        """
        if x.dim() != 1:
            raise ValueError("x.dim() must be 1")
        n_val = x.shape[0]
        if not mint.equal(mint.sort(x)[0], mint.arange(n_val)):
            raise ValueError("mint.equal(mint.sort(x)[0], mint.arange(n_val)) must be True")

    @staticmethod
    def are_permutations(x: ms.Tensor, dim=-1):
        """
        Checks if slices along the specified dimension in `x` are permutations of integers from 0 to N-1.

        Args:
            x (ms.Tensor): A tensor with any number of dimensions, containing slices of size N along `dim`.
            dim (int, optional): The dimension along which to check for permutations. Defaults to -1.
        """
        if x.dim() <= 0:
            raise ValueError("x.dim() must be greater than 0")

        n_val = x.shape[dim]
        # Create a view of x that moves the specified dimension to -1
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, n_val)
        for i in range(x.size(0)):
            Checker.is_permutation(x[i])

    @staticmethod
    def contains_identity(x: ms.Tensor, dim=-1):
        """
        Check if x contains the identity permutation

        Args:
            x (ms.Tensor): A tensor with any number of dimensions, containing slices of size N along `dim`.
            dim (int, optional): The dimension along which to check for permutations. Defaults to -1.
        """
        if x.dim() <= 0:
            raise ValueError("x.dim() must be greater than 0")

        n_val = x.shape[dim]
        # Create a view of x that moves the specified dimension to -1
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, n_val)
        expected = mint.arange(n_val).unsqueeze(dim=0)
        if not (x == expected).all(dim=-1).any():
            raise ValueError("(x == expected).all(dim=-1).any() must be True")

    @staticmethod
    def not_contain_identity(x: ms.Tensor, dim=-1):
        """
        Check if x does not contain the identity permutation

        Args:
            x (ms.Tensor): A tensor with any number of dimensions, containing slices of size N along `dim`.
            dim (int, optional): The dimension along which to check for permutations. Defaults to -1.
        """
        if x.dim() <= 0:
            raise ValueError("x.dim() must be greater than 0")

        n_val = x.shape[dim]
        # Create a view of x that moves the specified dimension to -1
        x = x.transpose(dim, -1).contiguous()
        x = x.reshape(-1, n_val)
        expected = mint.arange(n_val).unsqueeze(dim=0)
        if (x == expected).all(dim=-1).any():
            raise ValueError("(x == expected).all(dim=-1).any() must be False")

    @staticmethod
    def batch_permute(perm: ms.Tensor, x: ms.Tensor, x_permuted: ms.Tensor):
        """
        Args:
            perm (ms.Tensor):
                [..., N]
            x (ms.Tensor):
                [N, batch_dims_x]
            x_permuted (ms.Tensor):
                [..., N, batch_dims_x]
        """
        batch_shape = perm.shape[:-1]
        n_val = perm.shape[-1]
        if x.shape[0] != n_val:
            raise ValueError("x.shape[0] must be equal to n_val")
        perm = perm.view(-1, n_val)
        permuted_x = [x[perm[i]] for i in range(len(perm))]
        permuted_x = mint.stack(permuted_x, dim=0)  # [-1, N, batch_dims_x]
        target_shape = batch_shape + (n_val,) + x.shape[1:]
        if not mint.allclose(permuted_x.reshape(target_shape), x_permuted):
            raise ValueError("mint.allclose(permuted_x.reshape(target_shape), x_permuted) must be True")
