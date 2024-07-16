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

from mindspore import Tensor, nn, ops, Parameter, float32, float16
from mindspore.common.initializer import Normal, initializer

from ....e3.o3 import Irreps
from ....e3.utils import Ncon
from .layout import StridedLayout


class LinearInstruction:
    i_in: int
    i_out: int
    path_shape: tuple

    def __init__(self, i_in: int, i_out: int, path_shape: tuple):
        self.i_in = i_in
        self.i_out = i_out
        self.path_shape = path_shape


def _sum_tensors(xs: List[Tensor], shape, like: Tensor):
    if xs:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)


class Linear(nn.Cell):
    """Linear
    """

    # pylint: disable=R1710
    def __init__(
            self,
            irreps_in,
            irreps_out,
            instructions: Optional[List[Tuple[int, int]]] = None,
            pad_to_alignment: int = 1,
            dtype=float32,
    ):
        super().__init__()
        self.dtype = dtype

        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        # == Instructions ==
        if instructions is None:
            # By default, make all possible connections
            instructions = []
            for i_in, (_, ir_in) in enumerate(irreps_in):
                for i_out, (_, ir_out) in enumerate(irreps_out):
                    if ir_in == ir_out:
                        instructions.append((i_in, i_out))

            # note that "empty" instructions to/from empty irreps are dealt with in the codegen
        # Check if irreps can be strided
        try:
            layout_in = StridedLayout(irreps_in, pad_to_multiple=pad_to_alignment)
            layout_out = StridedLayout(irreps_out, pad_to_multiple=pad_to_alignment)
            # transfer to list (avoid ms bug)
            layout_in = [
                layout_in.irreps, layout_in.base_irreps, layout_in.pad_to_multiple, layout_in.dim, layout_in.base_dim,
                layout_in.mul
            ]
            layout_out = [
                layout_out.irreps, layout_out.base_irreps, layout_out.pad_to_multiple, layout_out.dim,
                layout_out.base_dim, layout_out.mul
            ]
        except ValueError as exc:
            # one cannot be strided
            raise ValueError('strided exception') from exc

        # group instructions by output
        ins_per_output = [[ins for ins in instructions if ins[1] == i] for i in range(len(layout_out[1]))]
        ins_group_irrep_slice: List[Tuple[int, int]] = []
        # check that each output is a mix of sequential irreps
        for ins_group in ins_per_output:
            # pylint: disable=C1801
            if len(ins_group) == 0:
                ins_group_irrep_slice.append(None)
                continue
            i_ins = set(ins[0] for ins in ins_group)
            ins_group_irrep_slice.append((min(i_ins), max(i_ins)))
            min_i_in, max_i_in = ins_group_irrep_slice[-1]
            if i_ins != set(range(min_i_in, 1 + max_i_in)):
                raise ValueError("wrong ins")
            if not all(ins[1] == ins_group[0][1] for ins in ins_group):
                raise ValueError("wrong ins")

        i_len = len(instructions)
        init_method = Normal(sigma=1.0)
        init_input = initializer(init_method, (layout_in[5] * layout_out[5] * i_len))
        linear_weights = Parameter(init_input)

        self.ins_per_output = ins_per_output
        self.ins_group_irrep_slice = ins_group_irrep_slice
        self.instructions = instructions
        self.layout_in = layout_in
        self.layout_out = layout_out

        layout_in_base_irreps_slice = []
        layout_in_base_irreps_dim = []

        for _, (_, ins_grp_ins) in enumerate(zip(ins_per_output, ins_group_irrep_slice)):
            layout_in_base_irreps_slice.append(
                (Irreps(layout_in[1].data[:ins_grp_ins[0]]).dim, Irreps(layout_in[1].data[:ins_grp_ins[1] + 1]).dim)
            )
            layout_in_base_irreps_dim.append(layout_in[1].data[ins_grp_ins[0]].ir.dim)

        count_len = [[] for _ in range(len(layout_out[1]))]
        layout_out_data_dim = []
        factor_list = []
        for i in range(len(count_len)):
            layout_out_data_dim.append(layout_out[1].data[i].dim)
            # pylint: disable=C1801
            if len(ins_per_output[i]) > 0:
                dividend = math.sqrt(sum(layout_in[5] for ins in instructions if ins[1] == i))
                if dividend != 0:
                    factor_list.append(1.0 / dividend)
                else:
                    raise ValueError("zero dividend")
            else:
                factor_list.append(1.0)

        self.layout_in_base_irreps_slice = layout_in_base_irreps_slice
        self.layout_in_base_irreps_dim = layout_in_base_irreps_dim
        self.layout_out_base_irreps = (layout_out[1].dim, len(layout_out[1]), layout_out_data_dim)
        self.factor_list = factor_list
        self.linear_weights = linear_weights
        self.ncon = Ncon([[-2, 1, 2], [-1, 1, 2, -3]])

    def construct(self, features):
        """construct
        """
        ins_per_output = self.ins_per_output
        ins_group_irrep_slice = self.ins_group_irrep_slice
        layout_in = self.layout_in
        layout_out = self.layout_out

        layout_in_base_irreps_slice = self.layout_in_base_irreps_slice
        layout_in_base_irreps_dim = self.layout_in_base_irreps_dim
        layout_out_base_irreps = self.layout_out_base_irreps
        factor_list = self.factor_list
        linear_weights = self.linear_weights

        # = Function definitions =
        x = features
        x = x.reshape(-1, layout_in[5], layout_in[4])
        ws = linear_weights

        outs = [[] for _ in range(layout_out_base_irreps[1])]

        w_index = 0

        for ins_grp_i, (ins_grp, ins_grp_ins) in enumerate(zip(ins_per_output, ins_group_irrep_slice)):
            # pylint: disable=C1801
            if len(ins_grp) == 0:
                continue
            # for a given group, which mixes a consecutive set of irreps of the same irrep,
            # we can reduce it to a rectangular operation:
            to_mix = x[:, :, layout_in_base_irreps_slice[ins_grp_i][0]:layout_in_base_irreps_slice[ins_grp_i][1]]
            # ^ has i index ranging over ins_grp_ins inputs *of same irrep*, so we can rectangularize with a new "n" dimension:
            n = 1 + ins_grp_ins[1] - ins_grp_ins[0]
            to_mix = to_mix.reshape(-1, layout_in[5], n, layout_in_base_irreps_dim[ins_grp_i])

            n_weight = int(layout_in[5]) * n
            n_weight = n_weight * int(layout_out[5])

            this_w = ws[w_index:w_index + n_weight].reshape(layout_out[5], layout_in[5], n)

            if self.dtype == float16:
                this_w = this_w.astype(float16)
                to_mix = to_mix.astype(float16)
            ncon_out = self.ncon([this_w, to_mix])
            if self.dtype == float16:
                ncon_out = ncon_out.astype(float32)
            outs[ins_grp[0][1]].append(ncon_out)
            w_index += n_weight

        outs = [
            _sum_tensors(
                o,
                shape=(
                    x.shape[0],
                    layout_out[5],
                    layout_out_base_irreps[2][i],
                ),
                like=x,
            ) * Tensor(factor_list[i]) for i, (ins_grp, o) in enumerate(zip(ins_per_output, outs))
        ]

        if len(outs) > 1:
            out = ops.cat(outs, axis=-1)
        else:
            out = outs[0]

        # pad output
        padding = layout_out[4] - layout_out_base_irreps[0]
        if padding > 0:
            out = ops.pad(
                out,
                (0, padding),
            )

        features = out

        return features
