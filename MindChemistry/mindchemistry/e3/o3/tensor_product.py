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
from mindspore import Tensor, nn, ops, Parameter, get_context, float32, int32, vmap
from mindspore.common.initializer import initializer
import mindspore as ms
from .irreps import Irreps
from .wigner import wigner_3j
from ..utils.ncon import Ncon
from ..utils.func import narrow
from ..utils.initializer import renormal_initializer
import numpy as np
from mindspore.numpy import tensordot


def _prod(x):
    out = 1
    for i in x:
        out *= i
    return out


sqrt = ops.Sqrt()
zeros = ops.Zeros()


def _sqrt(x, dtype=float32):
    """sqrt operator with producing a tensor"""
    return sqrt(Tensor(x, dtype=dtype))


def _sum_tensors(xs, shape, dtype):
    """sum tensors of same irrep."""
    if len(xs) > 0:
        out = xs[0].reshape(shape)
        for x in xs[1:]:
            out = out + x.reshape(shape)
        return out
    return zeros(shape, dtype)


def _compose(tensors, ir_data, instructions, batch_shape):
    """compose list of tensor `tensors` into a 1d-tensor by `ir_data`."""
    res = []
    for i_out, mir_out in enumerate(ir_data):
        if mir_out.mul > 0:
            res.append(_sum_tensors([out for ins, out in zip(instructions, tensors)
                                     if ins['i_out'] == i_out], shape=batch_shape + (mir_out.dim,),
                                    dtype=tensors[0].dtype))
    if len(res) > 1:
        res = ops.concat(res, axis=-1)
    else:
        res = res[0]
    return res


def _connect_init(irreps_in1, irreps_in2, irreps_out):
    """Input initial for 'connect' mode."""
    full_out = (irreps_in1 * irreps_in2).simplify()
    irreps_out = full_out if irreps_out is None else Irreps(irreps_out)

    instr = []
    for i_1, (_, ir_1) in enumerate(irreps_in1.data):
        for i_2, (_, ir_2) in enumerate(irreps_in2.data):
            ir_out_list = list(ir_1 * ir_2)
            for i_out, (_, ir_out) in enumerate(irreps_out.data):
                if ir_out in ir_out_list:
                    instr.append((i_1, i_2, i_out, 'uvw', True))

    return irreps_out, instr


def _full_init(irreps_in1, irreps_in2, irreps_out):
    """Input initial for 'full' mode."""
    full_out = irreps_in1 * irreps_in2
    irreps_out = full_out.filter(irreps_out)

    instr = []
    for i_1, (mul_1, ir_1) in enumerate(irreps_in1.data):
        for i_2, (mul_2, ir_2) in enumerate(irreps_in2.data):
            ir_out_list = list(ir_1 * ir_2)
            for i_out, (mul_out, ir_out) in enumerate(irreps_out.data):
                if ir_out in ir_out_list and mul_out == mul_1 * mul_2:
                    instr.append((i_1, i_2, i_out, 'uvuv', False))

    return irreps_out, instr


def _element_init(irreps_in1, irreps_in2, irreps_out):
    """Input initial for 'element' mode."""
    irreps_out = None if irreps_out is None else Irreps(irreps_out)

    if not irreps_in1.num_irreps == irreps_in2.num_irreps:
        raise ValueError(
            f"The total multiplicities of irreps_in1 {irreps_in1} and irreps_in2 {irreps_in2} should be equal.")

    irreps_in1_list = list(Irreps(irreps_in1).simplify().data)
    irreps_in2_list = list(Irreps(irreps_in2).simplify().data)

    i = 0
    while i < len(irreps_in1_list):
        mul_1, ir_1 = irreps_in1_list[i]
        mul_2, ir_2 = irreps_in2_list[i]

        if mul_1 < mul_2:
            irreps_in2_list[i] = (mul_1, ir_2)
            irreps_in2_list.insert(i + 1, (mul_2 - mul_1, ir_2))

        if mul_2 < mul_1:
            irreps_in1_list[i] = (mul_2, ir_1)
            irreps_in1_list.insert(i + 1, (mul_1 - mul_2, ir_1))
        i += 1

    out = []
    instr = []
    for i, ((mul, ir_1), (mul_2, ir_2)) in enumerate(zip(irreps_in1_list, irreps_in2_list)):
        for ir in ir_1 * ir_2:
            if irreps_out is not None and ir not in irreps_out:
                continue

            out.append((mul, ir))
            instr.append((i, i, len(out) - 1, 'uuu', False))

    return Irreps(irreps_in1_list), Irreps(irreps_in2_list), Irreps(out), instr


def _linear_init(irreps_in1, irreps_out):
    """Input initial for 'lnear' mode."""
    irreps_out = Irreps(irreps_out)

    instr = []
    for i_1, (_, ir_1) in enumerate(irreps_in1.data):
        for i_out, (_, ir_out) in enumerate(irreps_out.data):
            if ir_1 == ir_out:
                instr.append((i_1, 0, i_out, 'uvw', True))

    return irreps_out, instr


def _merge_init(irreps_in1, irreps_in2, irreps_out_filter):
    """Input initial for 'merge' mode."""
    irreps_out_filter = Irreps(
        irreps_out_filter) if irreps_out_filter is not None else irreps_in1 * irreps_in2

    irreps_out_list = []
    instr = []
    for i_1, (mul, ir_1) in enumerate(irreps_in1.data):
        for i_2, (_, ir_2) in enumerate(irreps_in2.data):
            for ir in ir_1 * ir_2:
                if ir in irreps_out_filter:
                    k = len(irreps_out_list)
                    irreps_out_list.append((mul, ir))
                    instr.append((i_1, i_2, k, 'uvu', True))

    irreps_out = Irreps(irreps_out_list)
    irreps_out, p, _ = irreps_out.sort()

    instr = [(i_1, i_2, p[i_out], mode, train)
             for i_1, i_2, i_out, mode, train in instr]

    return irreps_out, instr


def _raw_ins_check(mir_in1, mir_in2, mir_out, raw_ins):
    """Check raw input instructions."""
    if not mir_in1.ir.p * mir_in2.ir.p == mir_out.ir.p:
        raise ValueError(
            f"The parity of inputs and output do not match. \n \
                {mir_in1.ir.p} * {mir_in2.ir.p} should equal to {mir_out.ir.p}.")
    if not (abs(mir_in1.ir.l - mir_in2.ir.l) <= mir_out.ir.l and mir_out.ir.l <= mir_in1.ir.l + mir_in2.ir.l):
        raise ValueError(
            f"The degree of inputs and output do not match. \n \
                The degrees should be |{mir_in1.ir.l} - {mir_in2.ir.l}| <= {mir_out.ir.l} <= |{mir_in1.ir.l} + {mir_in2.ir.l}|.")
    if not raw_ins[3] in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']:
        raise ValueError(
            f"The connection mode should be in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv']")


def _mode_check(mul_in1, mul_in2, mul_out, ins):
    """Consistency check for multiplicities."""
    if ins['mode'] == 'uvw':
        if not ins['has_weight']:
            raise ValueError(f"The connection mode 'uvw' should have weights.")
    elif ins['mode'] == 'uuu':
        if not (mul_in1 == mul_in2 and mul_in2 == mul_out):
            raise ValueError(
                f"The multiplicity of inputs and output do not match. \
                    It should be {mul_in1} == {mul_in2} == {mul_out}.")
    elif ins['mode'] == 'uuw':
        if not mul_in1 == mul_in2:
            raise ValueError(
                f"The multiplicity of inputs do not match. \
                    It should be {mul_in1} == {mul_in2}.")
        if not (ins['has_weight'] or mul_out == 1):
            raise ValueError(
                f"The multiplicity of input or 'has_weight' do not match. \
                    If 'has_weight' == Flase, {mul_out} should equal to 1.")
    elif ins['mode'] == 'uvu':
        if not mul_in1 == mul_out:
            raise ValueError(
                f"The multiplicity of input 1 and output do not match. \
                    It should be {mul_in1} == {mul_out}.")
    elif ins['mode'] == 'uvv':
        if not mul_in2 == mul_out:
            raise ValueError(
                f"The multiplicity of input 2 and output do not match. \
                    It should be {mul_in2} == {mul_out}.")
    elif ins['mode'] == 'uvuv':
        if not mul_in1 * mul_in2 == mul_out:
            raise ValueError(
                f"The multiplicity of inputs and output do not match. \
                    It should be {mul_in1} * {mul_in2} == {mul_out}.")


def _init_einsum(mode, ls):
    """tensor graph contractions"""
    if mode == 'uuu':
        einsum = ops.Einsum("ijk,zui,zuj->zuk")
    elif mode == 'uuw':
        einsum = ops.Einsum("ijk,zui,zuj->zk")
    elif mode == 'uvu':
        einsum = ops.Einsum("ijk,zui,zvj->zuk")
    elif mode == 'uvv':
        einsum = ops.Einsum("ijk,zui,zvj->zvk")
    elif mode == 'uvuv':
        einsum = ops.Einsum("ijk,zui,zvj->zuvk")
    return einsum


def _init_einsum_weight(mode, weight_mode, ls):
    """tensor graph contractions with weights"""
    z = "z" if weight_mode == 'custom' else ""
    if mode == 'uvw':
        einsum = ops.Einsum(f"ijk,zui,zvj,{z}uvw->zwk")
    elif mode == 'uuu':
        einsum = ops.Einsum(f"ijk,zui,zuj,{z}u->zuk")
    elif mode == 'uuw':
        einsum = ops.Einsum(f"ijk,zui,zuj,{z}uw->zwk")
    elif mode == 'uvu':
        einsum = ops.Einsum(f"ijk,zui,zvj,{z}uv->zuk")
    elif mode == 'uvv':
        einsum = ops.Einsum(f"ijk,zui,zvj,{z}uv->zvk")
    elif mode == 'uvuv':
        einsum = ops.Einsum(f"ijk,zui,zvj,{z}uv->zuvk")
    return einsum


def _init_ncon(mode, ls):
    """tensor graph contractions"""
    if mode == 'uuu':
        con_list = [[1, 2, -3], [-1, -2, 1], [-1, -2, 2]]
    elif mode == 'uuw':
        con_list = [[1, 2, -2], [-1, 3, 1], [-1, 3, 2]]
    elif mode == 'uvu':
        con_list = [[1, 2, -3], [-1, -2, 1], [-1, 3, 2]]
    elif mode == 'uvv':
        con_list = [[1, 2, -3], [-1, 3, 1], [-1, -2, 2]]
    elif mode == 'uvuv':
        con_list = [[1, 2, -4], [-1, -2, 1], [-1, -3, 2]]
    ncon = Ncon(con_list)
    return ncon


class uvw_ncon_v2(nn.Cell):
    def __init__(self):
        super(uvw_ncon_v2, self).__init__()
        self.tensordot1 = tensordot
        self.tensordot2 = tensordot
        self.tensordot3 = vmap(tensordot, (0, 0, None), 0)

    def construct(self, m1, m2, m3, m4):
        temp1 = self.tensordot1(m3, m1, [2, 1])
        temp2 = self.tensordot1(m2, m4, [1, 0])
        res = self.tensordot3(temp2, temp1, ([0, 1], [1, 0]))
        return res


def _init_ncon_weight(mode, weight_mode, ls):
    """tensor graph contractions with weights"""
    if mode == 'uvw':
        con_list = [[1, 2, -3], [-1, 3, 1], [-1, 4, 2], [3, 4, -2]]
    elif mode == 'uuu':
        con_list = [[1, 2, -3], [-1, -2, 1], [-1, -2, 2], [-2]]
    elif mode == 'uuw':
        con_list = [[1, 2, -3], [-1, 3, 1], [-1, 3, 2], [3, -2]]
    elif mode == 'uvu':
        con_list = [[1, 2, -3], [-1, -2, 1], [-1, 3, 2], [-2, 3]]
    elif mode == 'uvv':
        con_list = [[1, 2, -3], [-1, 3, 1], [-1, -2, 2], [3, -2]]
    elif mode == 'uvuv':
        con_list = [[1, 2, -4], [-1, -2, 1], [-1, -3, 2], [-2, -3]]
    if weight_mode == 'custom':
        con_list[3] = [-1] + con_list[3]
    ncon = Ncon(con_list)
    return ncon


def _run_continue(ir1_data, ir2_data, irout_data, ins):
    """check trivial computations"""
    mir_in1 = ir1_data[ins['indice_one']]
    mir_in2 = ir2_data[ins['indice_two']]
    mir_out = irout_data[ins['i_out']]
    if mir_in1.dim == 0 or mir_in2.dim == 0 or mir_out.dim == 0:
        return True
    return False


class TensorProduct(nn.Cell):
    r"""
    Versatile tensor product operator of two input `Irreps` and a output `Irreps`, that sends two tensors into a tensor
    and keep the geometric tensor properties.
    This class integrates different typical usages: `TensorSquare`, `FullTensorProduct`, `FullyConnectedTensorProduct`,
    `ElementwiseTensorProduct` and `Linear`.

    A `TensorProduct` class defines an algebraic structure with equivariance.
    Ones the `TensorProduct` object is created and initialized, the algorithm is determined. For any given two legal input
    tensors, this object will provide a output tensor.
    If the object do not have learnable weights, the output tensor is deterministic.
    When the learnable weights are introduced, this operator will correspond to a general bilinear, equivariant operation,
    as a generalization of the standard tensor product.

    If `irreps_in2` is not specified, it will be assigned as `irreps_in1`, corresponding to `TensorSquare`.
    If `irreps_out` is not specified, this operator will account all possible output irreps.
    If both `irreps_out` and `instructions` are not specified, this operator is the standard tensor product without
    any learnable weights, corresponding to ``FullTensorProduct``.

    Each output irrep should satisfy:

    .. math::
        \| l_1 - l_2 \| \leq l_{out} \leq \| l_1 + l_2 \|
        p_1 p_2 = p_{out}

    Args:
        irreps_in1 (Union[str, Irrep, Irreps]): Irreps for the first input.

        irreps_in2 (Union[str, Irrep, Irreps, None]): Irreps for the second input. Default: None.
            If `irreps_in2` is None, `irreps_in2` will be assigned as '0e' in 'linear' instructions, or be assigned as `irreps_in1` in otherwise, corresponding to `TensorSquare`.

        irreps_out (Union[str, Irrep, Irreps, None]): Irreps for the output in 'connect' and custom instructions, or filter irreps for the output in otherwise.
            If `irreps_out` is None, `irreps_out` will be the full tensor product irreps (including all possible paths). Default: None.

        instructions (Union[str, List[Tule[int, int, int, str, bool, (float)]]]): List of tensor product path instructions. Default: 'full'.
            For `str` in {'full', 'connect', 'element', 'linear', 'mearge'}, the instructions are constructed automatically according to the different modes:

                - 'full': each output irrep for every pair of input irreps — is created and returned independently. The outputs are not mixed with each other.
                  Corresponding to the standard tensor product `FullTensorProduct` if `irreps_out` is not specified.
                - 'connect': each output is a learned weighted sum of compatible paths. This allows the operator to produce outputs with any multiplicity.
                  Corresponding to `FullyConnectedTensorProduct`.
                - 'element': the irreps are multiplied one-by-one. The inputs will be split and that the multiplicities of the outputs match with the multiplicities of the input.
                  Corresponding to `ElementwiseTensorProduct`.
                - 'linear': linear operation equivariant on the first irreps, while the second irreps is set to be '0e'. This can be regarded as the geometric tensors version of teh dense layer.
                  Corresponding to `Linear`.
                - 'merge': Automatically build 'uvu' mode instructions with trainable parameters. The `irreps_out` here plays the role of output filters.

            For `List[Tule[int, int, int, str, bool, (float)]]`, the instructions are constructed manually.
                Each instruction contain a tuple: (indice_one, indice_two, i_out, mode, has_weight, (optional: path_weight)).
                Each instruction puts ``in1[indice_one]`` :math:`\otimes` ``in2[indice_two]`` into ``out[i_out]``.

                 - `indice_one`, `indice_two`, `i_out`: int, the index of the irrep in irreps for `irreps_in1`, `irreps_in2` and `irreps_out` correspondingly.
                 - `mode`: str in {'uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv'}, the way of the multiplicities of each path are treated. 'uvw' is the fully mixed mode.
                 - `has_weight`: bool, `True` if this path should have learnable weights, otherwise `False`.
                 - `path_weight`:float, a multiplicative weight to apply to the output of this path. Defaults: 1.0.

        irrep_norm (str): {'component', 'norm'}, the assumed normalization of the input and output representations. Default: 'component'. Default: 'component'.

             - 'norm': :math:` \| x \| = \| y \| = 1 \Longrightarrow \| x \otimes y \| = 1`

        path_norm (str): {'element', 'path'}, the normalization method of path weights. Default: 'element'.

             - 'element': each output is normalized by the total number of elements (independently of their paths).
             - 'path': each path is normalized by the total number of elements in the path, then each output is normalized by the number of paths.

        weight_init (str): {'zeros', 'ones', 'truncatedNormal', 'normal', 'uniform', 'he_uniform', 'he_normal', 'xavier_uniform'}, the initial method of weights. Default: 'normal'.
        weight_mode (str): {'inner', 'share', 'custom'} determine the weights' mode. Default: 'inner'.

             - 'inner': weights will initialized in the tensor product internally.
             - 'share': weights should given manually without batch dimension.
             - 'custom': weights should given manually with batch dimension.


    Raises:
        ValueError: If `irreps_out` is not legal.
        ValueError: If the connection mode is not in ['uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv'].
        ValueError: If the degree of inputs and output do not match.
        ValueError: If the parity of inputs and output do not match.
        ValueError: If the multiplicity of inputs and output do not match.
        ValueError: If the connection mode is 'uvw', but `has_weight` is `False`.
        ValueError: If the connection mode is 'uuw' and `has_weight` is `False`, but the multiplicity is not equal to 1.
        ValueError: If the initial method is not supported.
        ValueError: If the number of input tensors is not match to the number of input irreps.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    Examples:
        Standard tensor product:

        >>> tp1 = TensorProduct('2x1o+4x0o', '1x1o+3x0e')
        TensorProduct [full] (2x1o+4x0o x 1x1o+3x0e -> 2x0e+12x0o+6x1o+2x1e+4x1e+2x2e)
        >>> v1 = ms.Tensor(np.linspace(1., 2., tp1.irreps_in1.dim), dtype=ms.float32)
        >>> v2 = ms.Tensor(np.linspace(2., 3., tp1.irreps_in2.dim), dtype=ms.float32)
        >>> tp1(v1, v2).shape
        (1, 60)

        Elementwise tensor product:

        >>> tp2 = TensorProduct('2x2e+4x1o', '3x1e+3x0o')
        TensorProduct [element] (2x2e+1x1o+3x1o x 2x1e+1x1e+3x0o -> 2x1e+2x2e+2x3e+1x0o+1x1o+1x2o+3x1e)
        >>> tp2.instructions
        [(0, 0, 0, 'uuu', False), (0, 0, 1, 'uuu', False), (0, 0, 2, 'uuu', False), (1, 1, 3, 'uuu', False),
        (1, 1, 4, 'uuu', False), (1, 1, 5, 'uuu', False), (2, 2, 6, 'uuu', False)]

        Custom tensor product with learnable weights:

        >>> tp3 = TensorProduct(
        ...     '3x2o+2x1o', '2x2e+4x1o+5x0e', '2x3o+8x1e+10x1o',
        ...     [
        ...         (0,0,0,'uvv',True),
        ...         (1,0,0,'uuu',True),
        ...         (1,1,1,'uvuv',True),
        ...         (1,2,2,'uvw',True)
        ...     ]
        ... )
        TensorProduct [custom] (3x2o+2x1o x 2x2e+4x1o+5x0e -> 2x3o+8x1e+10x1o)
        >>> [w.shape for w in tp3.weights]
        [(3, 2), (2,), (2, 4), (2, 5, 10)]

        Linear operation with an output filter:

        >>> tp4 = TensorProduct('2x1o', irreps_out='5x2e+4x1e+7x1o', instructions='connect')
        TensorProduct [linear] (2x2e+3x1o+3x0e x 1x0e -> 3x2e+5x1o+2x0e)
        >>> v1 = ms.Tensor(np.linspace(1., 2., tp.irreps_in1.dim), dtype=ms.float32)
        >>> tp4(v1).shape
        (1, 32)

    """
    __slots__ = ('irreps_in1', 'irreps_in2', 'irreps_out',
                 'weights', '_in2_is_none', '_mode', '_device', 'output_mask', 'core_mode')

    def __init__(
            self,
            irreps_in1,
            irreps_in2=None,
            irreps_out=None,
            instructions='full',
            dtype=float32,
            irrep_norm='component',
            path_norm='element',
            weight_init='normal',
            weight_mode='inner',
            core_mode='ncon',
            ncon_dtype=float32
    ):
        super().__init__()

        if weight_mode not in ['inner', 'share', 'custom']:
            raise ValueError(
                f"`weight_mode` should be one of ['inner', 'share', 'custom'].")
        if core_mode not in ['ncon', 'einsum']:
            raise ValueError(
                f"`core_mode` should be one of ['ncon', 'einsum'].")
        elif core_mode == 'einsum' and get_context('device_target') != 'GPU':
            raise ValueError(
                f"The `core_mode`: einsum only support GPU, but got {get_context('device_target')}.")
        self.weight_mode = weight_mode
        self.dtype = dtype
        self.core_mode = core_mode
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()

        self.irreps_in1 = Irreps(irreps_in1).simplify()
        if irreps_in2 is None:
            self.irreps_in2 = Irreps(irreps_in1).simplify()
            self._in2_is_none = True
        else:
            self.irreps_in2 = Irreps(irreps_in2).simplify()
            self._in2_is_none = False

        self.irreps_out, instructions = self._input_init(
            self.irreps_in1, self.irreps_in2, irreps_out, instructions)

        self.instr, self._ncons = self._ins_init(instructions)

        self.weight_numel = sum(_prod(ins['path_shape'])
                                for ins in self.instr if ins['has_weight'])

        self.weights = self._weight_init(weight_init)

        self.output_mask = self._init_mask()

        self._normalization(irrep_norm=irrep_norm, path_norm=path_norm)

        self.ncon_dtype = ncon_dtype

    def construct(self, v1, v2=None, weight=None):
        """Implement tensor product for input tensors."""
        self._weight_check(weight)

        if self._in2_is_none:
            if v2 is not None:
                raise ValueError(f"This tensor product should input 1 tensor.")

            if self._mode == 'linear':
                v2_shape = v1.shape[:-1] + (1,)
                v2 = self.ones(v2_shape, v1.dtype)
            else:
                v2 = v1.copy()
        else:
            if v2 is None:
                raise ValueError(
                    f"This tensor product should input 2 tensors.")
            if self._mode == 'linear':
                v2_shape = v1.shape[:-1] + (1,)
                v2 = self.ones(v2_shape, v1.dtype)

        batch_shape = v1.shape[:-1]
        v1s = self.irreps_in1.decompose(v1, batch=True)
        v2s = self.irreps_in2.decompose(v2, batch=True)
        weight = self._get_weights(weight)
        if not (v1.shape[-1] == self.irreps_in1.dim and v2.shape[-1] == self.irreps_in2.dim):
            raise ValueError(f"The shape of input tensors do not match.")

        v3_list = []
        weight_ind = 0
        fn = 0

        for ins in self.instr:
            if _run_continue(self.irreps_in1.data, self.irreps_in2.data, self.irreps_out.data, ins):
                continue
            fn = self._ncons[ins['i_ncon']]
            if ins['has_weight']:
                l = _prod(ins['path_shape'])
                w = narrow(weight, -1, weight_ind, l).reshape(((-1,)
                                                               if self.weight_mode == 'custom' else ()) + ins[
                                                                  'path_shape']).astype(self.ncon_dtype)
                weight_ind += l
                if self.core_mode == 'einsum':
                    v3 = fn((ins['wigner_matrix'].astype(self.ncon_dtype),
                             v1s[ins['indice_one']].astype(self.ncon_dtype),
                             v2s[ins['indice_two']].astype(self.ncon_dtype), w))
                else:
                    v3 = fn(
                        [ins['wigner_matrix'].astype(self.ncon_dtype), v1s[ins['indice_one']].astype(self.ncon_dtype),
                         v2s[ins['indice_two']].astype(self.ncon_dtype), w])
            else:
                if self.core_mode == 'einsum':
                    v3 = fn((ins['wigner_matrix'].astype(self.ncon_dtype),
                             v1s[ins['indice_one']].astype(self.ncon_dtype),
                             v2s[ins['indice_two']].astype(self.ncon_dtype)))
                else:
                    v3 = fn(
                        [ins['wigner_matrix'].astype(self.ncon_dtype), v1s[ins['indice_one']].astype(self.ncon_dtype),
                         v2s[ins['indice_two']].astype(self.ncon_dtype)])
            v3_list.append(ins['path_weight'].astype(self.dtype) * v3.astype(self.dtype))

        v_out = _compose(v3_list, self.irreps_out.data, self.instr, batch_shape)
        return v_out

    def __repr__(self):
        return f'TensorProduct [{self._mode}] ({self.irreps_in1.simplify().__repr__()} x {self.irreps_in2.simplify().__repr__()} -> {self.irreps_out.simplify().__repr__()} | {self.weight_numel} weights)'

    @property
    def instructions(self):
        return [tuple(ins.values())[:5] for ins in self.instr]

    def _input_init(self, irreps_in1, irreps_in2, irreps_out, instructions):
        if not isinstance(instructions, str):
            irreps_out = irreps_in1 * \
                         irreps_in2 if irreps_out is None else Irreps(irreps_out)
            self._mode = 'custom'
        else:
            if instructions == 'connect':
                irreps_out, instructions = _connect_init(
                    irreps_in1, irreps_in2, irreps_out)
                self._mode = 'connect'

            elif instructions == 'full':
                irreps_out, instructions = _full_init(
                    irreps_in1, irreps_in2, irreps_out)
                self._mode = 'full'

            elif instructions == 'element':
                self.irreps_in1, self.irreps_in2, irreps_out, instructions = _element_init(
                    irreps_in1, irreps_in2, irreps_out)
                self._mode = 'element'

            elif instructions == 'linear':
                self.irreps_in2 = Irreps('0e')
                irreps_out, instructions = _linear_init(irreps_in1, irreps_out)
                self._mode = 'linear'

            elif instructions == 'merge':
                irreps_out, instructions = _merge_init(
                    irreps_in1, irreps_in2, irreps_out)
                self._mode = 'merge'

            else:
                raise ValueError(
                    f"Unexpected instructions mode {instructions}")

        return irreps_out, instructions

    def _ins_init(self, raw_ins):
        """reform instructions"""
        raw_ins = [x if len(x) == 6 else x + (1.0,) for x in raw_ins]
        res = []
        ncons = []

        for ins in raw_ins:
            indice_one = ins[0]
            indice_two = ins[1]
            i_out = ins[2]
            mode = ins[3]
            has_weight = ins[4]
            path_weight = ins[5]

            mirs = (
                self.irreps_in1.data[indice_one], self.irreps_in2.data[indice_two], self.irreps_out.data[i_out])
            muls = (mirs[0].mul, mirs[1].mul, mirs[2].mul)

            _raw_ins_check(*mirs, ins)

            path_shape = {
                'uvw': (muls[0], muls[1], muls[2]),
                'uvu': (muls[0], muls[1]),
                'uvv': (muls[0], muls[1]),
                'uuw': (muls[0], muls[2]),
                'uuu': (muls[0],),
                'uvuv': (muls[0], muls[1]),
            }[mode]

            num_elements = {
                'uvw': (muls[0] * muls[1]),
                'uvu': muls[1],
                'uvv': muls[0],
                'uuw': muls[0],
                'uuu': 1,
                'uvuv': 1,
            }[mode]

            ls = (mirs[0].ir.l, mirs[1].ir.l, mirs[2].ir.l)

            d, op = self._ins_dict(indice_one, indice_two, i_out, mode, has_weight,
                                   path_weight, path_shape, num_elements, wigner_3j(*ls, self.dtype), ls)
            ncons.append(op)
            d['i_ncon'] = len(ncons) - 1
            res.append(d)

            _mode_check(*muls, res[-1])

        return res, ncons

    def _ins_dict(self, *args):
        """generate reformed instructions"""
        d = {}
        keys = ['indice_one', 'indice_two', 'i_out', 'mode', 'has_weight',
                'path_weight', 'path_shape', 'num_elements', 'wigner_matrix', 'ls']
        for i, arg in enumerate(args):
            d[keys[i]] = arg

        if d['has_weight']:
            if self.core_mode == 'einsum':
                operator = _init_einsum_weight(
                    d['mode'], self.weight_mode, d['ls'])
            else:
                operator = _init_ncon_weight(
                    d['mode'], self.weight_mode, d['ls'])
        else:
            if self.core_mode == 'einsum':
                operator = _init_einsum(d['mode'], d['ls'])
            else:
                operator = _init_ncon(d['mode'], d['ls'])

        return d, operator

    def _weight_init(self, init_method):
        """init weights"""
        init_method = renormal_initializer(init_method)

        if self.weight_numel > 0 and self.weight_mode == 'inner':
            weights = Parameter(
                initializer(init_method, (1, self.weight_numel), dtype=self.dtype).init_data().flatten())
        else:
            weights = None

        return weights

    def _init_mask(self):
        if self.irreps_out.dim > 0:
            output_mask = ops.cat([
                self.ones(mul * ir.dim, int32)
                if any(
                    (ins['i_out'] == i_out) and (ins['path_weight']
                                                 != 0) and (0 not in ins['path_shape'])
                    for ins in self.instr
                )
                else self.zeros(mul * ir.dim, int32)
                for i_out, (mul, ir) in enumerate(self.irreps_out.data)
            ])
        else:
            output_mask = Tensor(0)

        return output_mask

    def _normalization(self, irrep_norm, path_norm):
        """path normalization"""
        for ins in self.instr:
            mir_in1 = self.irreps_in1.data[ins['indice_one']]
            mir_in2 = self.irreps_in2.data[ins['indice_two']]
            mir_out = self.irreps_out.data[ins['i_out']]

            alpha = 1.
            if irrep_norm == 'component':
                alpha = mir_out.ir.dim
            if irrep_norm == 'norm':
                alpha = mir_in1.ir.dim * mir_in2.ir.dim

            x = 1.
            if path_norm == 'element':
                x = sum(i['num_elements']
                        for i in self.instr if i['i_out'] == ins['i_out'])
            if path_norm == 'path':
                x = ins['num_elements']
                x *= len([i for i in self.instr if i['i_out']
                          == ins['i_out']])

            if x > 0.0:
                alpha /= x

            alpha *= ins['path_weight']
            ins['path_weight'] = _sqrt(alpha, self.dtype)

    def _weight_check(self, weight):
        if self.weight_mode == 'inner':
            if weight is None:
                return True
            raise ValueError(
                f"For `weight_mode` {self.weight_mode}, the `weight` should not given manually.")
        elif self.weight_mode == 'share':
            if weight is None:
                raise ValueError(
                    f"For `weight_mode` {self.weight_mode}, the `weight` should given manually.")
            if not weight.ndim == 1:
                raise ValueError(
                    f"The shape of custom weight {weight.shape} is illegal.")
        elif self.weight_mode == 'custom':
            if weight is None:
                raise ValueError(
                    f"For `weight_mode` {self.weight_mode}, the `weight` should given manually.")
            if not weight.ndim > 1:
                raise ValueError(
                    f"Custom weight {weight} should have batch dimension if `weight_mode` is `'custom'`.")
        else:
            raise ValueError(f"Unknown `weight_mode`: {self.weight_mode}.")
        return True

    def _get_weights(self, weight):
        if weight is None:
            return self.weights
        else:
            return weight.reshape(-1, self.weight_numel)
