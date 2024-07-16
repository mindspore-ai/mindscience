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
# ==============================================================================
"""init"""

import math
from typing import List, Optional, Tuple

import numpy as np
import scipy as sp
from mindspore import Tensor, nn, float32, float16

from ....e3.o3 import Irreps, wigner_3j
from ....e3.utils import Ncon
from .layout import StridedLayout


class Instruction:
    """Instruction"""
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple

    def __init__(
            self, i_in1: int, i_in2: int, i_out: int, connection_mode: str, has_weight: bool, path_weight: float,
            path_shape: tuple
    ):
        self.i_in1 = i_in1
        self.i_in2 = i_in2
        self.i_out = i_out
        self.connection_mode = connection_mode
        self.has_weight = has_weight
        self.path_weight = path_weight
        self.path_shape = path_shape


class Contractor(nn.Cell):
    """Contractor"""

    def __init__(
            self,
            irreps_in1: Irreps,
            irreps_in2: Irreps,
            irreps_out: Irreps,
            instr: List[Tuple[int, int, int]],
            has_weight: bool,
            connection_mode: str,
            pad_to_alignment: int = 1,
            # pylint: disable=W0613
            shared_weights: bool = False,
            sparse_mode: Optional[str] = None,
            normalization: str = "component",
            dtype=float32,
    ):
        super().__init__()
        self.dtype = dtype

        in1_var = [1.0 for _ in irreps_in1]
        in2_var = [1.0 for _ in irreps_in2]
        out_var = [1.0 for _ in irreps_out]

        instructions = [
            Instruction(
                i_in1,
                i_in2,
                i_out,
                connection_mode,
                has_weight,
                1.0,
                {
                    "uvw": (
                        irreps_in1.data[i_in1].mul,
                        irreps_in2.data[i_in2].mul,
                        irreps_out.data[i_out].mul,
                    ),
                    "uvu": (irreps_in1.data[i_in1].mul, irreps_in2.data[i_in2].mul),
                    "uvv": (irreps_in1.data[i_in1].mul, irreps_in2.data[i_in2].mul),
                    "uuw": (irreps_in1.data[i_in1].mul, irreps_out.data[i_out].mul),
                    "uuu": (irreps_in1.data[i_in1].mul,),
                    "uvuv": (
                        irreps_in1.data[i_in1].mul,
                        irreps_in2.data[i_in2].mul,
                    ),
                }.get(connection_mode, None),
            ) for i_in1, i_in2, i_out in instr
        ]
        try:
            layout_in1 = StridedLayout(irreps_in1, pad_to_multiple=pad_to_alignment)
            layout_in2 = StridedLayout(irreps_in2, pad_to_multiple=pad_to_alignment)
            layout_out = StridedLayout(irreps_out, pad_to_multiple=pad_to_alignment)
        except ValueError as exc:
            raise ValueError('strided exception') from exc

        has_weight = instructions[0].has_weight
        if not all(ins.has_weight == has_weight for ins in instructions):
            raise ValueError("has_weight exception")
        if not has_weight:
            if connection_mode != "uuu":
                raise ValueError("wrong connection_mode")

        # Make the big w3j
        w3j_index = []
        w3j_values = []

        self.handle_w3j(
            connection_mode, normalization, in1_var, in2_var, out_var, instructions, layout_in1, layout_in2, layout_out,
            w3j_index, w3j_values
        )

        num_paths = len(instructions) if has_weight else 1

        w3j = sp.sparse.coo_array(
            (np.concatenate(w3j_values, axis=0), np.concatenate(w3j_index, axis=0).transpose()), (
                num_paths * layout_out.base_dim,
                layout_in1.base_dim * layout_in2.base_dim,
            )
        )
        w3j_i_indexes = np.floor_divide(w3j.col, layout_in1.base_dim)
        w3j_j_indexes = w3j.col % layout_in1.base_dim
        w3j_is_ij_diagonal = (layout_in1.base_dim == layout_in2.base_dim) and np.all(w3j_i_indexes == w3j_j_indexes)
        if w3j_is_ij_diagonal:
            w3j = sp.sparse.coo_array(
                (w3j.data, np.stack((w3j.row, w3j_i_indexes))), (
                    num_paths * layout_out.base_dim,
                    layout_in1.base_dim,
                )
            )

        w3j = self.check_sparse(sparse_mode, layout_in1, layout_in2, layout_out, num_paths, w3j, w3j_is_ij_diagonal)

        self.w3j = Tensor.from_numpy(w3j)
        self.w3j_is_ij_diagonal = w3j_is_ij_diagonal

        if w3j_is_ij_diagonal:
            self.ncon1 = Ncon([[-1, -2, -3], [-1, -2, -3]])
            self.ncon2 = Ncon([[-1, -2, 1], [-3, 1]])

        else:
            self.ncon1 = Ncon([[-1, -2, 1], [-3, 1, -4]])
            self.ncon2 = Ncon([[-1, -2, -3, 1], [-1, -2, 1]])

    def check_sparse(self, sparse_mode, layout_in1, layout_in2, layout_out, num_paths, w3j, w3j_is_ij_diagonal):
        """check_sparse"""
        if sparse_mode is None:
            if w3j_is_ij_diagonal:
                kij_shape = (
                    layout_out.base_dim,
                    layout_in1.base_dim,
                )
            else:
                kij_shape = (
                    layout_out.base_dim,
                    layout_in1.base_dim,
                    layout_in2.base_dim,
                )
            w3j = (w3j.toarray().reshape(((num_paths,) if num_paths > 1 else tuple()) + kij_shape))
            del kij_shape
        else:
            raise ValueError
        return w3j

    def handle_w3j(
            self, connection_mode, normalization, in1_var, in2_var, out_var, instructions, layout_in1, layout_in2,
            layout_out, w3j_index, w3j_values
    ):
        """handle_w3j"""
        for ins_i, ins in enumerate(instructions):
            mul_ir_in1 = layout_in1.base_irreps.data[ins.i_in1]
            mul_ir_in2 = layout_in2.base_irreps.data[ins.i_in2]
            mul_ir_out = layout_out.base_irreps.data[ins.i_out]

            if mul_ir_in1.ir.p * mul_ir_in2.ir.p != mul_ir_out.ir.p:
                raise ValueError("wrong irreps")
            # pylint: disable=C0325
            if not (abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l):
                raise ValueError("wrong irreps")

            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                raise ValueError

            this_w3j = wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l).asnumpy()
            this_w3j_index = this_w3j.nonzero()
            this_w3j_index = np.stack(np.array(this_w3j_index), axis=1)  ###
            w3j_values.append(this_w3j[this_w3j_index[:, 0], this_w3j_index[:, 1], this_w3j_index[:, 2]])

            # Normalize the path through its w3j entries
            if normalization == "component":
                w3j_norm_term = 2 * mul_ir_out.ir.l + 1
            if normalization == "norm":
                w3j_norm_term = (2 * mul_ir_in1.ir.l + 1) * (2 * mul_ir_in2.ir.l + 1)
            dividend = sum(
                in1_var[i.i_in1] * in2_var[i.i_in2] * {
                    "uvw": (layout_in1.mul * layout_in2.mul),
                    "uvu": layout_in2.mul,
                    "uvv": layout_in1.mul,
                    "uuw": layout_in1.mul,
                    "uuu": 1,
                    "uvuv": 1,
                }.get(connection_mode, None) for i in instructions if i.i_out == ins.i_out
            )
            if dividend != 0:
                alpha = math.sqrt(
                    ins.path_weight  # per-path weight
                    * out_var[ins.i_out]  # enforce output variance
                    * w3j_norm_term / dividend
                )
            else:
                raise ValueError
            w3j_values[-1] = w3j_values[-1] * alpha

            this_w3j_index[:, 0] += Irreps(layout_in1.base_irreps.data[:ins.i_in1]).dim
            this_w3j_index[:, 1] += Irreps(layout_in2.base_irreps.data[:ins.i_in2]).dim
            this_w3j_index[:, 2] += Irreps(layout_out.base_irreps.data[:ins.i_out]).dim
            # Now need to flatten the index to be for [pk][ij]
            w3j_index.append(
                np.concatenate(
                    (
                        (ins_i if ins.has_weight else 0) * layout_out.base_dim +
                        np.expand_dims(this_w3j_index[:, 2], axis=-1),
                        np.expand_dims(this_w3j_index[:, 0], axis=-1) * layout_in2.base_dim +
                        np.expand_dims(this_w3j_index[:, 1], axis=-1),
                    ),
                    axis=1,
                )
            )

    def construct(self, features, local_env_per_edge):
        """construct
        """
        if self.dtype == float16:
            if self.w3j_is_ij_diagonal:
                tmp = self.ncon1([features.astype(float16), local_env_per_edge.astype(float16)]).astype(float32)
                features = self.ncon2([tmp.astype(float16), self.w3j.astype(float16)]).astype(float32)
            else:
                tmp = self.ncon1([features.astype(float16), self.w3j.astype(float16)]).astype(float32)
                features = self.ncon2([tmp.astype(float16), local_env_per_edge.astype(float16)]).astype(float32)
        else:
            if self.w3j_is_ij_diagonal:
                tmp = self.ncon1([features, local_env_per_edge])
                features = self.ncon2([tmp, self.w3j])
            else:
                tmp = self.ncon1([features, self.w3j])
                features = self.ncon2([tmp, local_env_per_edge])
        return features
