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
from mindspore import nn, ops, vmap, jit_class
from mindspore.numpy import tensordot
from mindchemistry.e3 import o3
from mindchemistry.e3.o3 import Irreps

from .wigner import wigner_D


class SO3Embedding(nn.Cell):
    """
    SO3Embedding class
    """

    def __init__(self):
        self.embedding = None

    def _rotate(self, so3rotation, lmax_list, max_list):
        """
        SO3Embedding rotate
        """
        embedding_rotate = so3rotation[0].rotate(self.embedding, lmax_list[0],
                                                 max_list[0])
        self.embedding = embedding_rotate

    def _rotate_inv(self, so3rotation):
        """
        SO3Embedding rotate inverse
        """
        embedding_rotate = so3rotation[0].rotate_inv(self.embedding,
                                                     self.lmax_list[0],
                                                     self.mmax_list[0])
        self.embedding = embedding_rotate


@jit_class
class SO3Rotation:
    """
    SO3_Rotation class
    """

    def __init__(self, lmax, irreps_in, irreps_out):
        self.lmax = lmax
        self.irreps_in1 = Irreps(irreps_in)
        self.irreps_out = Irreps(irreps_out)
        self.tensordot_vmap = vmap(tensordot, (0, 0, None), 0)

    @staticmethod
    def narrow(inputs, axis, start, length):
        """
        SO3_Rotation narrow class
        """
        begins = [0] * inputs.ndim
        begins[axis] = start

        sizes = [i for i in inputs.shape]

        sizes[axis] = length
        res = ops.slice(inputs, begins, sizes)
        return res

    @staticmethod
    def rotation_to_wigner_d_matrix(edge_rot_mat, start_lmax, end_lmax):
        """
        SO3_Rotation rotation_to_wigner_d_matrix
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
        SO3_Rotation set_wigner
        """
        wigner = self.rotation_to_wigner_d_matrix(rot_mat3x3, 0, self.lmax)
        wigner_inv = []
        length = len(wigner)
        for i in range(length):
            wigner_inv.append(ops.swapaxes(wigner[i], 1, 2))
        return tuple(wigner), tuple(wigner_inv)

    def rotate(self, embedding, wigner):
        """
        SO3_Rotation rotate
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
        SO3_Rotation rotate_inv
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
