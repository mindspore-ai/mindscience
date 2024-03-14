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
"""gate"""
from mindspore import nn, ops, float32

from .activation import Activation
from ..o3.irreps import Irreps
from ..o3.tensor_product import TensorProduct
from ..utils.func import narrow


class _Extract(nn.Cell):
    """Extract tuple of tensors from irreps_in by irreps_outs with respecting instructions."""

    def __init__(self, irreps_in, irreps_outs, instructions):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_outs = tuple(Irreps(irreps) for irreps in irreps_outs)
        self.instr = instructions

        if not len(self.irreps_outs) == len(self.instr):
            raise ValueError('inputs are illegal')
        for irreps_out, ins in zip(self.irreps_outs, self.instr):
            if not len(irreps_out) == len(ins):
                raise ValueError('inputs are illegal')

    def construct(self, x):
        """construct"""
        out = []
        for i in range(len(self.irreps_outs)):
            if self.instr[i] == tuple(range(len(self.irreps_in.data))):
                out.append(x)
            else:
                out_i = []
                for i_in in self.instr[i]:
                    out_i.append(narrow(x, -1, *self.irreps_in.slice_tuples[i_in]))
                if out_i:
                    out.append(ops.concat(out_i, -1))
        return out


class _Sortcut(nn.Cell):
    """Sort and cut a tensor by irreps_outs."""

    def __init__(self, *irreps_outs):
        super().__init__()
        self.irreps_outs = tuple(Irreps(irreps).simplify() for irreps in irreps_outs)
        irreps_in = sum(self.irreps_outs, Irreps([]))

        i = 0
        instructions = []
        for irreps_out in self.irreps_outs:
            instructions.append(tuple(range(i, i + len(irreps_out))))
            i += len(irreps_out)

        irreps_in, p, _ = irreps_in.sort()
        instructions = [tuple(p[i] for i in x) for x in instructions]

        self.cut = _Extract(irreps_in, self.irreps_outs, instructions)
        self.irreps_in = irreps_in.simplify()

    def construct(self, x):
        return self.cut(x)


class Gate(nn.Cell):
    r"""
    Gate activation function. The input contain three parts: the first part `irreps_scalars` are scalars that only be
    affected by activation functions `acts`;
    the second part `irreps_gates` are scalars that be affected by activation functions `act_gates` and be multiplied
    on the third part.

    .. math::
        \left(\bigoplus_i \phi_i(x_i) \right) \oplus \left(\bigoplus_j \phi_j(g_j) y_j \right)

    where :math:`x_i` and :math:`\phi_i` are from `irreps_scalars` and `acts`, and :math:`g_j`, :math:`\phi_j`,
    and :math:`y_j` are from `irreps_gates`, `act_gates`, and `irreps_gated`.

    Args:
        irreps_scalars (Union[str, Irrep, Irreps]): the input scalar irreps that will be passed through the
        activation functions `acts`.
        acts (List[Func]): a list of activation functions for each part of `irreps_scalars`.
            The length of the `acts` will be clipped or filled by identity functions to match the length of
            `irreps_scalars`.
        irreps_gates (Union[str, Irrep, Irreps]): the input scalar irreps that will be passed through the activation
        functions `act_gates` and multiplied by `irreps_gated`.
        act_gates (List[Func]): a list of activation functions for each part of `irreps_gates`.
            The length of the `acts` will be clipped or filled by identity functions to match the length of
            `irreps_gates`.
        irreps_gated (Union[str, Irrep, Irreps]): the input irreps that will be gated.

    Raises:
        ValueError: If `irreps_scalars` or `irreps_gates` contain non-scalar irrep.
        ValueError: If the total multiplication of `irreps_gates` do not match the total multiplication of
        `irreps_gated`.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> Gate('2x0e', [ops.tanh], '1x0o+2x0e', [ops.abs], '2x1o+1x2e')
        Gate (2x0e+1x0o+2x0e+2x1o+1x2e -> 2x0e+2x1o+1x2e)
    """

    def __init__(self, irreps_scalars, acts, irreps_gates, act_gates, irreps_gated, dtype=float32, ncon_dtype=float32):
        super().__init__()
        irreps_scalars = Irreps(irreps_scalars)
        irreps_gates = Irreps(irreps_gates)
        irreps_gated = Irreps(irreps_gated)

        # pylint: disable=C1801
        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}")
        # pylint: disable=C1801
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}")
        if not irreps_gates.num_irreps == irreps_gated.num_irreps:
            raise ValueError(f"There are {irreps_gated.num_irreps} irreps in irreps_gated, \
                    but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates")

        self.sc = _Sortcut(irreps_scalars, irreps_gates, irreps_gated)
        self.irreps_scalars, self.irreps_gates, self.irreps_gated = self.sc.irreps_outs

        if self.irreps_scalars.num_irreps == 0:
            self._has_scalar = False
        else:
            self._has_scalar = True
            self.act_pass = Activation(irreps_scalars, acts, dtype=dtype)
            irreps_scalars = self.act_pass.irreps_out
        self.act_gates = Activation(irreps_gates, act_gates, dtype=dtype)
        irreps_gates = self.act_gates.irreps_out

        self.tp = TensorProduct(irreps_gated, irreps_gates, instructions='element', dtype=dtype, ncon_dtype=ncon_dtype)
        irreps_gated = self.tp.irreps_out

        self.irreps_in = self.sc.irreps_in
        self.irreps_out = irreps_scalars + irreps_gated

    def construct(self, x):
        """Implement the gate activation function for the input tensor."""

        scalars, gates, gated = self.sc(x)
        if self._has_scalar:
            scalars = self.act_pass(scalars)

        if gates.shape[-1] > 0:
            gates = self.act_gates(gates)
            gated = self.tp(gated, gates)
            if self._has_scalar:
                x = ops.concat([scalars, gated], axis=-1)
            else:
                x = gated
        else:
            x = scalars

        return x

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"
