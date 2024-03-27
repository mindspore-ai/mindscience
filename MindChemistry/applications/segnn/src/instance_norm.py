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
Instance normalization
"""
from mindspore import nn, ops, Parameter
from mindchemistry.e3.o3 import Irreps
from mindchemistry.graph.graph import AggregateNodeToGlobal


class InstanceNorm(nn.Cell):
    '''Instance normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.
    Parameters
    ----------
    irreps : `Irreps`
        representation
    eps : float
        avoid division by zero when we normalize by the variance
    affine : bool
        do we have weight and bias parameters
    reduce : {'mean', 'sum'}
        method used to reduce
    normalization : {'norm' or 'component'}
        method used to normalization
    dtype : {float16, float32, float64}
    '''

    def __init__(self, irreps, eps=1e-5, affine=True, reduce='mean', normalization='component', dtype=None):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        self.ones = ops.Ones()
        self.zeros = ops.Zeros()

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0)
        num_features = self.irreps.num_irreps
        if affine:
            self.weight = Parameter(self.ones(num_features, dtype))
            self.bias = Parameter(self.zeros(num_scalar, dtype))
        else:
            self.weight = None
            self.bias = None

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ['mean', 'sum'], "reduce needs to be 'mean' or 'sum'"
        self.reduce = reduce

        self.scatter = AggregateNodeToGlobal(reduce)

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def construct(self, x_input, batch, batch_size, node_mask):
        """construct

        Args:
            x_input (Tensor): x_input
            batch (Tensor): batch
            batch_size: batch size
            node_mask: mask for node

        Returns:
            output: Tensor
        """
        dim = x_input.shape[-1]
        fields = []
        ix = 0
        iw = 0
        ib = 0
        for item in self.irreps.data:
            mul, ir = item.mul, item.ir
            d = ir.dim
            field = x_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            size = batch_size
            # For scalars first compute and subtract the mean
            if ir.l == 0:
                field = field.reshape(-1, mul)
                field_mean = self.scatter(field, batch, dim_size=size, mask=node_mask)
                field = field - field_mean[batch]
                field = field.reshape(-1, mul, d)

            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))

            if self.reduce == 'mean' or self.reduce == 'sum':
                field_norm = self.scatter(field_norm, batch, dim_size=size, mask=node_mask)
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm[batch].reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            if self.affine and d == 1:  # scalars
                bias = self.bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            field = field.reshape(-1, mul * d)  # [batch * sample, mul * repr]
            # Finally, make a mask
            field = field * node_mask.reshape(-1, 1)

            # Save the result, to be stacked later with the rest
            fields.append(field)

        if ix != dim:
            fmt = "`ix` should have reached x_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = ops.concat(fields, axis=-1)  # [batch * sample, stacked features]
        return output
