# Copyright 2024 Huawei Technologies Co., Ltd
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
so3 file
"""
import mindspore as ms
from mindspore import ops, vmap, jit_class
from mindspore.numpy import tensordot
from .. import o3
from ..o3 import Irreps

from .wigner import wigner_D


@jit_class
class SO3Rotation:
    """
    Class for handling SO(3) rotations of spherical-harmonic irreps.

    Args:
        lmax (int): Maximum angular momentum to be considered.
        irreps_in (Union[str, Irreps]): Input irreps specification.
        irreps_out (Union[str, Irreps]): Output irreps specification.

    Examples:
        >>> from mindscience.e3nn.so2_conv import SO3Rotation
        >>> rot = SO3Rotation(lmax=2, irreps_in="1x0e + 1x1o", irreps_out="1x1o")
        >>> wigner, wigner_inv = rot.set_wigner(rot_mat3x3)
        >>> rotated = rot.rotate(embedding, wigner)
    """

    def __init__(self, lmax, irreps_in, irreps_out):
        self.lmax = lmax
        self.irreps_in1 = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.tensordot_vmap = vmap(tensordot, (0, 0, None), 0)

    @staticmethod
    def narrow(inputs, axis, start, length):
        """
        Narrow (slice) a tensor along a specified axis.

        Args:
            inputs (Tensor): The tensor to be sliced.
            axis (int): The axis along which to perform the slice.
            start (int): The starting index of the slice.
            length (int): The number of elements to include in the slice.

        Returns
            Tensor, The sliced tensor.
        """
        begins = [0] * inputs.ndim
        begins[axis] = start

        sizes = list(inputs.shape)

        sizes[axis] = length
        res = ops.slice(inputs, begins, sizes)
        return res

    @staticmethod
    def rotation_to_wigner_d_matrix(edge_rot_mat, start_lmax, end_lmax):
        """
        Convert a batch of :math:`3 \times 3` rotation matrices into Wigner-D matrices for the
        specified range of angular momenta.

        Args:
            edge_rot_mat (Tensor): Batch of SO(3) rotation matrices of shape (..., 3, 3).
            start_lmax (int): Minimum angular momentum to include.
            end_lmax (int): Maximum angular momentum to include.

        Returns:
            list[Tensor], List of Wigner-D matrices for l = start_lmax … end_lmax, each of shape (..., 2l+1, 2l+1).
        """
        x = edge_rot_mat @ ms.Tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        rvalue = (ops.swapaxes(
            o3.angles_to_matrix(alpha, beta, ops.zeros_like(alpha)), -1, -2)
                  @ edge_rot_mat)
        gamma = ops.atan2(rvalue[..., 0, 2], rvalue[..., 0, 0])

        block_list = []
        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma).astype(ms.float32)
            block_list.append(block)
        return block_list

    def set_wigner(self, rot_mat3x3):
        """
        Compute Wigner-D matrices and their inverses from a batch of :math:`3 \times 3` rotation matrices.

        Args:
            rot_mat3x3 (Tensor): Batch of SO(3) rotation matrices of shape (..., 3, 3).

        Returns:
            tuple[list[Tensor], list[Tensor]], A tuple containing two lists.

            - wigner: List of Wigner-D matrices for l = 0 … lmax, each of shape (..., 2l+1, 2l+1).
            - wigner_inv: List of transposed (inverse) Wigner-D matrices for l = 0 … lmax, same shapes.
        """
        wigner = self.rotation_to_wigner_d_matrix(rot_mat3x3, 0, self.lmax)
        wigner_inv = []
        length = len(wigner)
        for i in range(length):
            wigner_inv.append(ops.swapaxes(wigner[i], 1, 2))
        return tuple(wigner), tuple(wigner_inv)

    def rotate(self, embedding, wigner):
        """
        Rotate an embedding tensor according to the provided Wigner-D matrices.

        Args:
            embedding (Tensor): Input tensor of shape (..., irreps_in.dim) containing the spherical-harmonic
                coefficients to be rotated.
            wigner (tuple[Tensor]): Tuple of Wigner-D matrices for l = 0 … lmax, each of shape (..., 2l+1, 2l+1).

        Returns:
            tuple[Tensor], Tuple of rotated tensors, one per irrep in irreps_in, each of shape (..., mul, 2l+1).
        """
        res = []
        batch_shape = embedding.shape[:-1]
        for (s, l), mir in zip(self.irreps_in1.slice_tuples,
                               self.irreps_in1.data):
            v_slice = self.narrow(embedding, -1, s, l)
            if embedding.ndim == 1:
                res.append((v_slice.reshape((1,) + batch_shape +
                                            (mir.mul, mir.ir.dim)), mir.ir))
            else:
                res.append(
                    (v_slice.reshape(batch_shape + (mir.mul, mir.ir.dim)),
                     mir.ir))
        rotate_data_list = []
        for data, ir in res:
            self.tensordot_vmap(data.astype(ms.float16),
                                wigner[ir.l].astype(ms.float16), ([1], [1]))
            rotate_data = self.tensordot_vmap(data.astype(ms.float16),
                                              wigner[ir.l].astype(ms.float16),
                                              ((1), (1))).astype(ms.float32)
            rotate_data_list.append(rotate_data)
        return tuple(rotate_data_list)

    def rotate_inv(self, embedding, wigner_inv):
        """
        Apply the inverse SO(3) rotation to an embedding tensor using the provided inverse Wigner-D matrices.

        Args:
            embedding (tuple[Tensor]): Tuple of tensors, one per irrep in irreps_out, each of shape (..., mul, 2l+1).
            wigner_inv (tuple[Tensor]): Tuple of inverse (transposed) Wigner-D matrices for l = 0 … lmax,
                each of shape (..., 2l+1, 2l+1).

        Returns:
            Tensor, The rotated-back tensor of shape (..., irreps_out.dim) obtained by concatenating the
            inverse-rotated irreps.
        """
        res = []
        batch_shape = embedding[0].shape[0:1]
        index = 0
        for (_, _), mir in zip(self.irreps_out.slice_tuples,
                               self.irreps_out.data):
            v_slice = embedding[index]
            if embedding[0].ndim == 1:
                res.append((v_slice, mir.ir))
            else:
                res.append((v_slice, mir.ir))
            index = index + 1
        rotate_back_data_list = []
        for data, ir in res:
            rotate_back_data = self.tensordot_vmap(
                data.astype(ms.float16), wigner_inv[ir.l].astype(ms.float16),
                ((1), (1))).astype(ms.float32)
            rotate_back_data_list.append(
                rotate_back_data.view(batch_shape + (-1,)))
        return ops.cat(rotate_back_data_list, -1)
