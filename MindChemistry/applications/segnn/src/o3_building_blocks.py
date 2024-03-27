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
TensorProduct and Normalizition
"""
from math import sqrt
import numpy as np

from mindspore import nn, ops, Tensor, Parameter, int64
from mindchemistry.e3.o3 import Irreps, FullyConnectedTensorProduct
from mindchemistry.e3.nn import Gate


def prod(iter_data):
    res = 1
    for i in iter_data:
        res *= i
    return res


def weight_views(weights, instrs):
    offset = 0
    for ins in instrs:
        if ins['has_weight']:
            flatsize = prod(ins['path_shape'])
            this_weight = weights.narrow(-1, offset, flatsize).view(ins['path_shape'])
            offset += flatsize
            yield this_weight


class O3TensorProduct(nn.Cell):
    """ A bilinear layer, computing CG tensorproduct and normalising them.

    Parameters
    ----------
    irreps_in1 : o3.Irreps
        Input irreps.
    irreps_out : o3.Irreps
        Output irreps.
    irreps_in2 : o3.Irreps
        Second input irreps.
    dtype : [float16, float32, float64]
        type of float to use
    ncon_dtype : [float16, float32, float64]
        type of float for ncon
    tp_rescale : bool
        If true, rescales the tensor product.

    """

    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, dtype=None, ncon_dtype=None, tp_rescale=True):
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_out = irreps_out
        # Init irreps_in2
        if irreps_in2 is None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = irreps_in2

        self.dtype = dtype
        self.tp_rescale = tp_rescale
        # Build the layers
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            ncon_dtype=ncon_dtype,
            dtype=dtype
        )

        self.zeros = ops.Zeros()
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slice
        # Store tuples of slices and corresponding biases in a list
        self.biases_tmp = []
        self.biases_slice_idx = []
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_bias = self.zeros(self.irreps_out_dims[slice_idx], dtype)
                self.biases_tmp += [out_bias]
                self.biases_slice_idx += [slice_idx]
        self.slices_sqrt_k = {}
        self.tensor_product_init()
        # Adapt parameters so they can be applied using vector operations.
        self.vectorise()

    def tensor_product_init(self):
        """
        uniform weights and bias
        """
        slices_fan_in = {}
        for weight, instr in zip(weight_views(self.tp.weights, self.tp.instr), self.tp.instructions):
            slice_idx = instr[2]
            mul_1, mul_2, _ = weight.shape
            fan_in = mul_1 * mul_2
            slices_fan_in[slice_idx] = (
                slices_fan_in[slice_idx] + fan_in if slice_idx in slices_fan_in.keys() else fan_in
            )

        # Do the initialization of the weights in each instruction
        offset = 0
        for weight, instr in zip(weight_views(self.tp.weights, self.tp.instr), self.tp.instructions):
            slice_idx = instr[2]
            if self.tp_rescale:
                sqrt_k = 1 / sqrt(slices_fan_in[slice_idx])
            else:
                sqrt_k = 1.
            flat_size = prod(weight.shape)

            self.tp.weights[offset: offset+flat_size] = ops.uniform((flat_size,), Tensor(-sqrt_k), Tensor(sqrt_k))

            offset += flat_size
            self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

        for (out_slice_idx, out_bias) in zip(self.biases_slice_idx, self.biases_tmp):
            sqrt_k = 1 / sqrt(slices_fan_in[out_slice_idx])
            out_bias[:] = ops.uniform(out_bias.shape, Tensor(-sqrt_k), Tensor(sqrt_k))

    def vectorise(self):
        """
        Adapts the bias parameter and the sqrt_k corrections so they can be applied
        using vectorised operations
        """
        bias_len = len(self.biases_tmp)
        if bias_len > 0:
            self.biases_tmp = ops.concat(self.biases_tmp, axis=0)
            self.biases = Parameter(self.biases_tmp, name='biases')

            bias_idx = Tensor([], dtype=int64)
            for slice_idx in range(len(self.irreps_out_orders)):
                if self.irreps_out_orders[slice_idx] == 0:
                    out_slice = self.irreps_out.slice[slice_idx]
                    bias_idx = ops.concat((bias_idx, Tensor(np.arange(out_slice.start, out_slice.stop), int64)), axis=0)
            self.bias_idx = bias_idx
        else:
            self.biases = None

        # Now onto the sqrt_k correction
        sqrt_k_correction = self.zeros(self.irreps_out.dim, self.dtype)
        for instr in self.tp.instructions:
            slice_idx = instr[2]
            slices, sqrt_k = self.slices_sqrt_k[slice_idx]
            sqrt_k_correction[slices] = sqrt_k
        self.sqrt_k_correction = sqrt_k_correction

    def forward_tp_rescale_bias(self, data_in1, data_in2, mask=None):
        """
        tp compute,rescale and bias
        """
        if data_in2 is None:
            data_in2 = ops.ones_like(data_in1[:, 0:1])

        data_out = self.tp(data_in1, data_in2)

        if self.tp_rescale:
            data_out /= self.sqrt_k_correction

        if self.biases_tmp is not None:
            data_out[:, self.bias_idx] += self.biases

        if mask is not None:
            data_out = data_out * mask.reshape(-1, 1)
        return data_out

    def construct(self, data_in1, data_in2=None, mask=None):
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2, mask)
        return data_out


class O3TensorProductSwishGate(O3TensorProduct):
    """
    TensorProduct with Gate
    """
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, dtype=None, ncon_dtype=None):
        # The first type is assumed to be scalar and passed through the activation
        irreps_g_scalars = Irreps(str(irreps_out.data[0]))
        # The remaining types are gated
        irreps_g_gate = Irreps("{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps('+'.join([str(irreps_out.data[i]) for i in range(1, len(irreps_out.data))]))

        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()

        # Build the layers
        super(O3TensorProductSwishGate, self).__init__(irreps_in1, irreps_g, irreps_in2, dtype, ncon_dtype)
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(
                irreps_g_scalars,
                [nn.SiLU()],
                irreps_g_gate,
                [ops.sigmoid],
                irreps_g_gated,
                dtype=dtype,
                ncon_dtype=ncon_dtype
            )
        else:
            self.gate = nn.SiLU()

    def construct(self, data_in1, data_in2=None, mask=None):
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2, mask)
        data_out = self.gate(data_out)

        return data_out
