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
e3modules
"""
from mindchemistry.e3.o3.irreps import Irreps
from mindchemistry.e3.o3.tensor_product import TensorProduct
from mindchemistry.e3.nn.gate import _Extract
from mindchemistry.e3.o3.sub import Linear
from mindchemistry.e3.o3.wigner import wigner_3j
from mindchemistry.graph.graph import AggregateNodeToGlobal, LiftGlobalToNode
from mindspore import Tensor, ops, Parameter, ParameterTuple, jit_class
import mindspore.nn as nn
import mindspore as ms
import numpy as np
from .utils import irreps_from_l1l2


class SkipConnection(nn.Cell):
    """
    SkipConnection class
    """

    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        self.sc = None

        if irreps_in == irreps_out:
            self.sc = None
        else:
            self.sc = Linear(irreps_in=irreps_in, irreps_out=irreps_out, ncon_dtype=ms.float16)

    def construct(self, old, new):
        """
        SkipConnection class construct process
        """
        if self.sc is not None:
            old = self.sc(old)

        return old + new


class E3LayerNorm(nn.Cell):
    """
    E3LayerNorm class
    """

    def __init__(self,
                 irreps_in,
                 eps=1e-5,
                 affine=True,
                 normalization='component',
                 subtract_mean=True,
                 divide_norm=False):
        super().__init__()

        self.irreps_in = [[mul, ir] for mul, ir in Irreps(irreps_in)]

        self.eps = eps

        if affine:
            ib, iw = 0, 0
            weight_slices, bias_slices = [], []
            for mul, ir in irreps_in:
                if ir.is_scalar():  # bias only to 0e
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = Parameter(ops.ones([iw]))
            self.bias = Parameter(ops.zeros([ib]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            raise ValueError(f'affine False')

        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        self.normalization = normalization
        self.reset_parameters()

        self.aggregate = AggregateNodeToGlobal(mode="mean")
        self.lift = LiftGlobalToNode(mode="multi_graph")
        self.mean = ops.ReduceMean(keep_dims=True)

    def reset_parameters(self):
        """
        reset parameter to default
        """
        if self.weight is not None:
            self.weight.data.fill(1)
        if self.bias is not None:
            self.bias.data.fill(0)

    def construct(self,
                  x: ms.Tensor,
                  batch: ms.Tensor = None,
                  mask_degree: ms.Tensor = None,
                  mask_dim1: ms.Tensor = None,
                  mask_dim3: ms.Tensor = None):
        """
        net construct process
        """
        if batch is None:
            batch = ops.full([x.shape[0]], 0, dtype=ms.int64)

        batch_size = 1

        out = []
        ix = 0
        for index, (mul, ir) in enumerate(self.irreps_in):
            field = x[:, ix:ix + mul * ir.dim].reshape(-1, mul, ir.dim)  # [node, mul, repr]
            if self.subtract_mean or ir.l == 0:
                if mask_degree is not None:
                    mean = self.aggregate(field, batch, dim_size=batch_size, mask=mask_dim1)
                else:
                    mean = self.aggregate(field, batch, dim_size=batch_size)
                mean = self.mean(mean, [1])
                mean = self.lift(mean, batch)
                field = field - mean
                if mask_degree is not None:
                    field = ops.mul(field, mask_dim3)

            if self.divide_norm or ir.l == 0:  # do not divide norm for l>0 irreps if subtract_mean=False
                if mask_degree is not None:
                    var = self.aggregate(ops.square(field), batch, dim_size=batch_size, mask=mask_dim1)
                else:
                    var = self.aggregate(ops.square(field), batch, dim_size=batch_size)
                var = self.mean(var, [1])
                if self.normalization == 'norm':
                    var = var * ir.dim
                std = ops.sqrt(var)
                std = self.lift(std, batch)
                field = ops.true_divide(field, std + self.eps)

                # affine
            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]

            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[index]]
                field = field + bias[None, :, None]

            #### for dynamic shape ###
            if mask_degree is not None:
                field = ops.mul(field, mask_dim3)

            out.append(field.reshape(-1, mul * ir.dim))
            ix += mul * ir.dim

        out = ops.cat(out, axis=-1)

        return out


class E3ElementWise(nn.Cell):
    """
    E3ElementWise class
    """

    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_in_for_construct = []
        for mul, ir in self.irreps_in:
            self.irreps_in_for_construct.append([mul, ir])
        len_weight = 0
        for mul, _ in self.irreps_in:
            len_weight += mul

        self.len_weight = len_weight

    def construct(self, x, weight):
        """
        E3ElementWise class construct process
        """
        # x should have shape [edge/node, channels]
        # weight should have shape [edge/node, self.len_weight]
        ix = 0
        iw = 0
        out = []
        for mul, ir in self.irreps_in_for_construct:
            field = x[:, ix:ix + mul * ir.dim]
            field = field.reshape(-1, mul, ir.dim)
            field = field * weight[:, iw:iw + mul][:, :, None]
            field = field.reshape(-1, mul * ir.dim)

            ix += mul * ir.dim
            iw += mul
            out.append(field)

        return ops.cat(out, axis=-1)


class SeparateWeightTensorProduct(nn.Cell):
    """
    SeparateWeightTensorProduct class
    """

    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        '''z_i = W'_{ij}x_j W''_{ik}y_k'''
        super().__init__()

        irreps_in1 = Irreps(irreps_in1).simplify()
        irreps_in2 = Irreps(irreps_in2).simplify()
        irreps_out = Irreps(irreps_out).simplify()

        instr_tp = []
        weights1, weights2 = [], []
        name_index = 0
        for i1, (mul1, ir1) in enumerate(irreps_in1):
            for i2, (mul2, ir2) in enumerate(irreps_in2):
                for i_out, (mul_out, ir3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2:
                        weights1_name = 'weights1' + str(name_index)
                        weights2_name = 'weights2' + str(name_index)

                        np.random.seed(42)
                        weights1.append(
                            Parameter(Tensor(np.random.randn(mul1, mul_out), dtype=ms.float32), name=weights1_name))
                        weights2.append(
                            Parameter(Tensor(np.random.randn(mul2, mul_out), dtype=ms.float32), name=weights2_name))

                        instr_tp.append((i1, i2, i_out, 'uvw', True, 1.0))
                        name_index = name_index + 1

        self.tp = TensorProduct(irreps_in1,
                                irreps_in2,
                                irreps_out,
                                instr_tp,
                                weight_mode='share',
                                ncon_dtype=ms.float16,
                                **kwargs)

        self.weights1 = ParameterTuple(weights1)
        self.weights2 = ParameterTuple(weights2)

    def construct(self, x1, x2):
        """
        SeparateWeightTensorProduct class construct process
        """
        weights = []

        for weight1, weight2 in zip(self.weights1, self.weights2):
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))

        weights = ops.cat(weights)
        res = self.tp(x1, x2, weights)
        return res


class SelfTp(nn.Cell):
    """
    SelfTp class
    """

    def __init__(self, irreps_in, irreps_out, **kwargs):
        '''z_i = W'_{ij}x_j W''_{ik}x_k (k>=j)'''
        super().__init__()

        irreps_in = Irreps(irreps_in).simplify()
        irreps_out = Irreps(irreps_out).simplify()

        instr_tp = []
        weights1, weights2 = [], []
        name_index = 0
        for i1, (mul1, ir1) in enumerate(irreps_in):
            for i2 in range(i1, len(irreps_in.data)):
                mul2, ir2 = irreps_in[i2]
                for i_out, (mul_out, ir3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2:
                        weights1_name = 'weights1' + str(name_index)
                        weights2_name = 'weights2' + str(name_index)

                        np.random.seed(42)
                        weights1.append(
                            Parameter(Tensor(np.random.randn(mul1, mul_out), dtype=ms.float32), name=weights1_name))
                        weights2.append(
                            Parameter(Tensor(np.random.randn(mul2, mul_out), dtype=ms.float32), name=weights2_name))

                        instr_tp.append((i1, i2, i_out, 'uvw', True, 1.0))
                        name_index = name_index + 1

        self.tp = TensorProduct(irreps_in,
                                irreps_in,
                                irreps_out,
                                instr_tp,
                                weight_mode='share',
                                ncon_dtype=ms.float16,
                                **kwargs)

        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)

    def construct(self, x):
        """
        SelfTp class construct process
        """
        weights = []

        for weight1, weight2 in zip(self.weights1, self.weights2):
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))

        weights = ops.cat(weights)
        res = self.tp(x, x, weights)
        return res


class RotateNet(nn.Cell):
    """
    RotateNet class
    """

    def __init__(self, default_dtype_ms, spinful=False):
        super(RotateNet, self).__init__()

        self.spinful = spinful

        # openmx的实球谐函数基组变复球谐函数
        self.us_openmx = {
            0:
                ms.Tensor([1], dtype=ms.complex64),
            1:
                ms.Tensor([[-1 / 1.4142135623730951, 1j / 1.4142135623730951, 0], [0, 0, 1],
                           [1 / 1.4142135623730951, 1j / 1.4142135623730951, 0]], dtype=ms.complex64),
            2:
                ms.Tensor([[0, 1 / 1.4142135623730951, -1j / 1.4142135623730951, 0, 0],
                           [0, 0, 0, -1 / 1.4142135623730951, 1j / 1.4142135623730951], [1, 0, 0, 0, 0],
                           [0, 0, 0, 1 / 1.4142135623730951, 1j / 1.4142135623730951],
                           [0, 1 / 1.4142135623730951, 1j / 1.4142135623730951, 0, 0]],
                          dtype=ms.complex64),
            3:
                ms.Tensor([[0, 0, 0, 0, 0, -1 / 1.4142135623730951, 1j / 1.4142135623730951],
                           [0, 0, 0, 1 / 1.4142135623730951, -1j / 1.4142135623730951, 0, 0],
                           [0, -1 / 1.4142135623730951, 1j / 1.4142135623730951, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0],
                           [0, 1 / 1.4142135623730951, 1j / 1.4142135623730951, 0, 0, 0, 0],
                           [0, 0, 0, 1 / 1.4142135623730951, 1j / 1.4142135623730951, 0, 0],
                           [0, 0, 0, 0, 0, 1 / 1.4142135623730951, 1j / 1.4142135623730951]],
                          dtype=ms.complex64),
        }
        # openmx的实球谐函数基组变wiki的实球谐函数 https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        self.us_openmx2wiki = {
            0: ops.eye(1, dtype=default_dtype_ms),
            1: ops.eye(3, dtype=default_dtype_ms)[[1, 2, 0]],
            2: ops.eye(5, dtype=default_dtype_ms)[[2, 4, 0, 3, 1]],
            3: ops.eye(7, dtype=default_dtype_ms)[[6, 4, 2, 0, 1, 3, 5]]
        }

        self.us_wiki2openmx = {k: v.T for k, v in self.us_openmx2wiki.items()}

        self.dtype = default_dtype_ms

    def construct(self, h, l_left, l_right):
        """
        RotateNet class construct process
        """
        # wiki2openmx_H
        return self.us_openmx2wiki.get(l_left, None).T @ h @ self.us_openmx2wiki.get(l_right, None)


@jit_class
class SortIrreps:
    """
    SortIrreps jit class
    """

    def __init__(self, irreps_in):
        irreps_in = Irreps(irreps_in)
        sorted_irreps = irreps_in.sort()

        irreps_out_list = [((mul, ir),) for mul, ir in sorted_irreps.irreps]
        instructions = [(i,) for i in sorted_irreps.inv]
        self.extr = _Extract(irreps_in, irreps_out_list, instructions)

        irreps_in_list = [((mul, ir),) for mul, ir in irreps_in]
        instructions_inv = [(i,) for i in sorted_irreps.p]
        self.extr_inv = _Extract(sorted_irreps.irreps, irreps_in_list, instructions_inv)

        self.irreps_in = irreps_in
        self.irreps_out = sorted_irreps.irreps.simplify()

    def forward(self, x):
        r'''irreps_in -> irreps_out'''
        extracted = self.extr(x)
        return ops.cat(extracted, axis=-1)

    def inverse(self, x):
        r'''irreps_out -> irreps_in'''
        extracted_inv = self.extr_inv(x)
        return ops.cat(extracted_inv, axis=-1)


class E3TensorDecompNet(nn.Cell):
    """
    E3TensorDecompNet class
    """

    def __init__(self, net_irreps_out, out_js_list, default_dtype_ms, spinful=False, no_parity=False, if_sort=False):
        super(E3TensorDecompNet, self).__init__()
        self.dtype = default_dtype_ms
        self.spinful = spinful

        self.out_js_list = out_js_list
        if net_irreps_out is not None:
            net_irreps_out = Irreps(net_irreps_out)

        required_irreps_out = Irreps(None)
        in_slices = [0]
        wms = []  # wm = wigner_multiplier
        h_slices = [0]
        wms_h = []

        for _, (h_l1, h_l2) in enumerate(out_js_list):

            mul = 1
            _, required_irreps_out_single, _ = irreps_from_l1l2(h_l1, h_l2, mul, spinful, no_parity=no_parity)
            required_irreps_out += required_irreps_out_single

            # spinful case, example: (1x0.5)x(2x0.5) = (1+2+3)x(0+1) = (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4)
            # everything on r.h.s. as a whole constitutes a slice in in_slices
            # each bracket on r.h.s. above corresponds to a slice in in_slice_sp
            # = construct slices =
            in_slices.append(required_irreps_out.dim)
            h_slices.append(h_slices[-1] + (2 * h_l1 + 1) * (2 * h_l2 + 1))

            # = get CG coefficients multiplier to act on net_out =
            wm = []
            wm_h = []
            for _, ir in required_irreps_out_single:
                for _ in range(mul):
                    test1 = wigner_3j(h_l1, h_l2, ir.l, dtype=default_dtype_ms)
                    wm.append(test1)
                    wm_h.append(wigner_3j(ir.l, h_l1, h_l2, dtype=default_dtype_ms) * (2 * ir.l + 1))

            wm = ops.cat(wm, axis=-1)
            wm_h = ops.cat(wm_h, axis=0)
            wms.append(wm)
            wms_h.append(wm_h)

        self.in_slices = in_slices

        self.wms = wms

        self.h_slices = h_slices
        self.wms_h = wms_h

        # = register rotate kernel =
        self.rotate_kernel = RotateNet(default_dtype_ms, spinful=spinful)

        self.sort = None
        if if_sort:
            self.sort = SortIrreps(required_irreps_out)

        if self.sort is not None:
            self.required_irreps_out = self.sort.irreps_out
        else:
            self.required_irreps_out = required_irreps_out

    def construct(self, net_out):
        """
        E3TensorDecompNet class construct process
        """
        if self.sort is not None:
            net_out = self.sort.inverse(net_out)
        out = []
        length = len(self.out_js_list)
        for i in range(length):
            in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
            net_out_block = net_out[:, in_slice]

            hblock = ops.sum(self.wms[i][None, :, :, :] * net_out_block[:, None, None, :], dim=-1)
            hblock = self.rotate_kernel(hblock, *self.out_js_list[i])
            out.append(hblock.reshape(net_out.shape[0], -1))
        return ops.cat(out, axis=-1)
