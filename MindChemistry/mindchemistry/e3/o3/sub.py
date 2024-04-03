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
"""sub"""
from typing import NamedTuple
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore import ops, float32
from .tensor_product import TensorProduct
from .irreps import Irreps
from ..utils.func import narrow


class FullyConnectedTensorProduct(TensorProduct):
    r"""
    Fully-connected weighted tensor product. All the possible path allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1
    + l_2` are made.
    Equivalent to `TensorProduct` with `instructions='connect'`. For details, see `mindchemistry.e3.TensorProduct`.

    Args:
        irreps_in1 (Union[str, Irrep, Irreps]): Irreps for the first input.
        irreps_in2 (Union[str, Irrep, Irreps]): Irreps for the second input.
        irreps_out (Union[str, Irrep, Irreps]): Irreps for the output.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations.
        Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal',
        'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> FullyConnectedTensorProduct('2x1o', '1x1o+3x0e', '5x2e+4x1o')
        TensorProduct [connect] (2x1o x 1x1o+3x0e -> 5x2e+4x1o)

    """

    def __init__(self,
                 irreps_in1,
                 irreps_in2,
                 irreps_out,
                 ncon_dtype=float32,
                 **kwargs):
        super().__init__(irreps_in1,
                         irreps_in2,
                         irreps_out,
                         instructions='connect',
                         ncon_dtype=ncon_dtype,
                         **kwargs)


class FullTensorProduct(TensorProduct):
    r"""
    Full tensor product between two irreps.

    Equivalent to `TensorProduct` with `instructions='full'`. For details, see `mindchemistry.e3.TensorProduct`.

    Args:
        irreps_in1 (Union[str, Irrep, Irreps]): Irreps for the first input.
        irreps_in2 (Union[str, Irrep, Irreps]): Irreps for the second input.
        filter_ir_out (Union[str, Irrep, Irreps, None]): Filter to select only specific `Irrep`
        of the output. Default: None.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations.
        Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal',
        'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> FullTensorProduct('2x1o+4x0o', '1x1o+3x0e')
        TensorProduct [full] (2x1o+4x0o x 1x1o+3x0e -> 2x0e+12x0o+6x1o+2x1e+4x1e+2x2e)

    """

    def __init__(self,
                 irreps_in1,
                 irreps_in2,
                 filter_ir_out=None,
                 ncon_dtype=float32,
                 **kwargs):
        super().__init__(irreps_in1,
                         irreps_in2,
                         filter_ir_out,
                         instructions='full',
                         ncon_dtype=ncon_dtype,
                         **kwargs)


class ElementwiseTensorProduct(TensorProduct):
    r"""
    Elementwise connected tensor product.

    Equivalent to `TensorProduct` with `instructions='element'`. For details, see `mindchemistry.e3.TensorProduct`.

    Args:
        irreps_in1 (Union[str, Irrep, Irreps]): Irreps for the first input.
        irreps_in2 (Union[str, Irrep, Irreps]): Irreps for the second input.
        filter_ir_out (Union[str, Irrep, Irreps, None]): Filter to select only specific `Irrep` of the output.
            Default: None.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations.
            Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal',
            'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> ElementwiseTensorProduct('2x2e+4x1o', '3x1e+3x0o')
        TensorProduct [element] (2x2e+1x1o+3x1o x 2x1e+1x1e+3x0o -> 2x1e+2x2e+2x3e+1x0o+1x1o+1x2o+3x1e)

    """

    def __init__(self,
                 irreps_in1,
                 irreps_in2,
                 filter_ir_out=None,
                 ncon_dtype=float32,
                 **kwargs):
        super().__init__(irreps_in1,
                         irreps_in2,
                         filter_ir_out,
                         instructions='element',
                         ncon_dtype=ncon_dtype,
                         **kwargs)


class Linear(TensorProduct):
    r"""
    Linear operation equivariant.

    Equivalent to `TensorProduct` with `instructions='linear'`. For details, see `mindchemistry.e3.TensorProduct`.

    Args:
        irreps_in (Union[str, Irrep, Irreps]): Irreps for the input.
        irreps_out (Union[str, Irrep, Irreps]): Irreps for the output.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations.
            Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal',
            'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> Linear('2x2e+3x1o+3x0e', '3x2e+5x1o+2x0e')
        TensorProduct [linear] (2x2e+3x1o+3x0e x 1x0e -> 3x2e+5x1o+2x0e)

    """

    def __init__(self, irreps_in, irreps_out, ncon_dtype=float32, **kwargs):
        super().__init__(irreps_in,
                         None,
                         irreps_out,
                         instructions='linear',
                         ncon_dtype=ncon_dtype,
                         **kwargs)


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


def _prod(x):
    out = 1
    for i in x:
        out *= i
    return out


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def _sum_tensors_withbias(xs, shape, dtype):
    """sum tensors of same irrep."""
    if xs:
        if len(xs[0].shape) == 1:
            out = xs[0]
        else:
            out = xs[0].reshape(shape)
        for x in xs[1:]:
            if len(x.shape) == 1:
                out = out + x
            else:
                out = out + x.reshape(shape)
        return out
    return ops.zeros(shape, dtype=dtype)


def _compose(tensors, ir_data, instructions, batch_shape):
    """compose list of tensor `tensors` into a 1d-tensor by `ir_data`."""
    res = []
    for i_out, mir_out in enumerate(ir_data):
        if mir_out.mul > 0:
            res.append(
                _sum_tensors_withbias([
                    out for ins, out in zip(instructions, tensors)
                    if ins['i_out'] == i_out
                ],
                                      shape=batch_shape + (mir_out.dim,),
                                      dtype=tensors[0].dtype))

    if len(res) > 1:
        res = ops.concat(res, axis=-1)
    else:
        res = res[0]
    return res


def _run_continue(ir1_data, ir2_data, irout_data, ins):
    """check trivial computations"""
    mir_in1 = ir1_data[ins['indice_one']]
    mir_in2 = ir2_data[ins['indice_two']]
    mir_out = irout_data[ins['i_out']]
    if mir_in1.dim == 0 or mir_in2.dim == 0 or mir_out.dim == 0:
        return True
    return False


class LinearBias(TensorProduct):
    r"""
    Linear operation equivariant with option to add bias.

    Equivalent to `TensorProduct` with `instructions='linear'` with option to add bias. For details,
    see `mindchemistry.e3.TensorProduct`.

    Args:
        irreps_in (Union[str, Irrep, Irreps]): Irreps for the input.
        irreps_out (Union[str, Irrep, Irreps]): Irreps for the output.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations.
        Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal',
        'xavier_uniform'}, the initial method of weights. Default: 'normal'.
        has_bias (bool): whether add bias to calculation

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> LinearBias('2x2e+3x1o+3x0e', '3x2e+5x1o+2x0e')
        TensorProduct [linear] (2x2e+3x1o+3x0e x 1x0e -> 3x2e+5x1o+2x0e)

    """

    def __init__(self,
                 irreps_in,
                 irreps_out,
                 has_bias,
                 ncon_dtype=float32,
                 **kwargs):
        super().__init__(irreps_in,
                         None,
                         irreps_out,
                         instructions='linear',
                         ncon_dtype=ncon_dtype,
                         **kwargs)
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)

        biases = [has_bias and ir.is_scalar() for _, ir in irreps_out]

        is_scalar_num = biases.count(True)

        instructions = [
            Instruction(i_in=-1,
                        i_out=i_out,
                        path_shape=(mul_ir.dim,),
                        path_weight=1.0)
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]
        self.has_bias = has_bias
        self.bias_numel = None
        self.bias_instructions = None
        if self.has_bias:
            self.bias_instructions = []
            for i_out, (bias, mul_ir) in enumerate(zip(biases, self.irreps_out)):
                if bias:
                    path_shape = (mul_ir.dim,)
                    path_weight = 1.0
                    instruction = Instruction(i_in=-1, i_out=i_out, path_shape=path_shape, path_weight=path_weight)
                    self.bias_instructions.append(instruction)

            if is_scalar_num == 1:
                self.bias_numel = sum(irreps_out.data[i.i_out].dim
                                      for i in instructions if i.i_in == -1)
                bias = ops.zeros((self.bias_numel))
                self.bias = Parameter(bias, name="bias")
                self.instr.append({
                    "i_out": self.bias_instructions[0].i_out,
                    "indice_one": self.bias_instructions[0].i_in
                })
            else:
                bias = ops.zeros((is_scalar_num, 1))
                self.bias = Parameter(bias, name="bias")

                for bias_instr in self.bias_instructions:
                    self.instr.append({
                        "i_out": bias_instr.i_out,
                        "indice_one": bias_instr.i_in
                    })

        self.bias_add = P.BiasAdd()
        self.ncon_dtype = ncon_dtype

    def construct(self, v1, v2=None, weight=None):
        """Implement tensor product for input tensors."""
        self._weight_check(weight)

        if self._in2_is_none:
            if v2 is not None:
                raise ValueError(f"This tensor product should input 1 tensor.")

            if self._mode == 'linear':
                v2_shape = v1.shape[:-1] + (1,)
                v2 = ops.ones(v2_shape, v1.dtype)
            else:
                v2 = v1.copy()
        else:
            if v2 is None:
                raise ValueError(
                    f"This tensor product should input 2 tensors.")
            if self._mode == 'linear':
                v2_shape = v1.shape[:-1] + (1,)
                v2 = ops.ones(v2_shape, v1.dtype)

        batch_shape = v1.shape[:-1]

        v2s = self.irreps_in2.decompose(v2, batch=True)
        v1s = self.irreps_in1.decompose(v1, batch=True)

        weight = self._get_weights(weight)

        if not (v1.shape[-1] == self.irreps_in1.dim
                and v2.shape[-1] == self.irreps_in2.dim):
            raise ValueError(f"The shape of input tensors do not match.")

        v3_list = []
        weight_ind = 0
        fn = 0
        index_one = 'indice_one'
        index_two = 'indice_two'
        index_wigner = 'wigner_matrix'

        for ins in self.instr:
            if ins[index_one] == -1 or _run_continue(self.irreps_in1.data,
                                                     self.irreps_in2.data,
                                                     self.irreps_out.data, ins):
                continue
            fn = self._ncons[ins['i_ncon']]
            if ins['has_weight']:
                l = _prod(ins['path_shape'])
                w = narrow(
                    weight, -1, weight_ind,
                    l).reshape((
                        (-1,) if self.weight_mode == 'custom' else ()) +
                               ins['path_shape']).astype(self.ncon_dtype)
                weight_ind += l
                if self.core_mode == 'einsum':
                    v3 = fn((ins[index_wigner].astype(self.ncon_dtype),
                             v1s[ins[index_one]].astype(self.ncon_dtype),
                             v2s[ins[index_two]].astype(self.ncon_dtype), w))
                else:
                    v3 = fn([
                        ins[index_wigner].astype(self.ncon_dtype),
                        v1s[ins[index_one]].astype(self.ncon_dtype),
                        v2s[ins[index_two]].astype(self.ncon_dtype), w
                    ])
            else:
                if self.core_mode == 'einsum':
                    v3 = fn((ins[index_wigner].astype(self.ncon_dtype),
                             v1s[ins[index_one]].astype(self.ncon_dtype),
                             v2s[ins[index_two]].astype(self.ncon_dtype)))
                else:
                    v3 = fn([
                        ins[index_wigner].astype(self.ncon_dtype),
                        v1s[ins[index_one]].astype(self.ncon_dtype),
                        v2s[ins[index_two]].astype(self.ncon_dtype)
                    ])
            v3_list.append(ins['path_weight'].astype(self.dtype) *
                           v3.astype(self.dtype))

        if self.has_bias:
            if len(self.bias_instructions) == 1:
                v3_list.append(self.bias)
            else:
                for i in range(len(self.bias_instructions)):
                    v3_list.append(self.bias[i])

        v_out = _compose(v3_list, self.irreps_out.data, self.instr,
                         batch_shape)

        return v_out


class TensorSquare(TensorProduct):
    r"""
    Compute the square tensor product of a tensor.

    Equivalent to `TensorProduct` with `irreps_in2=None and instructions='full' or 'connect'`. For details,
    see `mindchemistry.e3.TensorProduct`.

    If `irreps_out` is given, this operation is fully connected.
    If `irreps_out` is not given, the operation has no parameter and is like full tensor product.

    Args:
        irreps_in (Union[str, Irrep, Irreps]): Irreps for the input.
        irreps_out (Union[str, Irrep, Irreps, None]): Irreps for the output. Default: None.
        filter_ir_out (Union[str, Irrep, Irreps, None]): Filter to select only specific `Irrep`
        of the output. Default: None.
        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations.
        Default: 'component'. Default: 'component'.
        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.
        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal',
        'xavier_uniform'}, the initial method of weights. Default: 'normal'.

    Raises:
        ValueError: If both `irreps_out` and `filter_ir_out` are not None.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        >>> TensorSquare('2x1o', irreps_out='5x2e+4x1e+7x1o')
        TensorProduct [connect] (2x1o x 2x1o -> 5x2e+4x1e)
        >>> TensorSquare('2x1o+3x0e', filter_ir_out='5x2o+4x1e+2x0e')
        TensorProduct [full] (2x1o+3x0e x 2x1o+3x0e -> 4x0e+9x0e+4x1e)

    """

    def __init__(self,
                 irreps_in,
                 irreps_out=None,
                 filter_ir_out=None,
                 ncon_dtype=float32,
                 **kwargs):
        if irreps_out is None:
            super().__init__(irreps_in,
                             None,
                             filter_ir_out,
                             instructions='full',
                             ncon_dtype=ncon_dtype,
                             **kwargs)
        else:
            if filter_ir_out is None:
                super().__init__(irreps_in,
                                 None,
                                 irreps_out,
                                 instructions='connect',
                                 ncon_dtype=ncon_dtype,
                                 **kwargs)
            else:
                raise ValueError(
                    "Both `irreps_out` and `filter_ir_out` are not None, this is ambiguous."
                )
