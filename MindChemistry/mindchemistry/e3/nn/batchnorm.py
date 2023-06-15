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

from mindspore import nn, Parameter, ops, float32

from ..o3.irreps import Irreps


class BatchNorm(nn.Cell):
    r"""
    Batch normalization for orthonormal representations.
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Args:
        irreps (Union[str, Irrep, Irreps]): the input irreps.
        eps (float): avoid division by zero when we normalize by the variance. Default: 1e-5.
        momentum (float): momentum of the running average. Default: 0.1.
        affine (bool): do we have weight and bias parameters. Default: True.
        reduce (str): {'mean', 'max'}, method used to reduce. Default: 'mean'.
        instance (bool): apply instance norm instead of batch norm. Default: Flase.
        normalization (str): {'component', 'norm'}, normalization method. Default: 'component'.

    Raises:
        ValueError: If `reduce` is not in ['mean', 'max'].
        ValueError: If `normalization` is not in ['component', 'norm'].

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    """

    def __init__(self, irreps, eps=1e-5, momentum=0.1, affine=True, reduce='mean', instance=False,
                 normalization='component', dtype=float32):
        super().__init__()
        self.irreps = Irreps(irreps)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.instance = instance
        self.reduce = reduce
        self.normalization = normalization
        self.training = True

        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps

        self.running_mean = None if self.instance else Parameter(ops.zeros(num_scalar, dtype=dtype),
                                                                 requires_grad=False)
        self.running_var = None if self.instance else Parameter(ops.ones(num_features, dtype=dtype),
                                                                requires_grad=False)

        self.weight = Parameter(ops.ones(num_features, dtype=dtype)) if affine else None
        self.bias = Parameter(ops.zeros(num_scalar, dtype=dtype)) if affine else None

    def _roll_avg(self, curr, update):
        return (1 - self.momentum) * curr + self.momentum * update

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def construct(self, inputs):
        inputs_shape = inputs.shape
        batch = inputs_shape[0]
        dim = inputs_shape[-1]
        inputs = inputs.reshape(batch, -1, dim)

        new_means = []
        new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mir in self.irreps.data:
            mul = mir.mul
            ir = mir.ir

            d = ir.dim
            field = inputs[:, :, ix: ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            # (batch, sample, mul, repr)
            field = field.reshape(batch, -1, mul, d)

            if ir.is_scalar():  # scalars
                if self.training or self.instance:
                    if self.instance:
                        field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    else:
                        field_mean = field.mean([0, 1]).reshape(mul)  # [mul]
                        new_means.append(
                            self._roll_avg(self.running_mean[irm:irm + mul], field_mean)
                        )
                else:
                    field_mean = self.running_mean[irm: irm + mul]
                irm += mul

                # (batch, sample, mul, repr)
                field = field - field_mean.reshape(-1, 1, mul, 1)

            if self.training or self.instance:
                if self.normalization == 'norm':
                    field_norm = field.pow(2).sum(3)  # [batch, sample, mul]
                elif self.normalization == 'component':
                    field_norm = field.pow(2).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError(f"Invalid normalization option {self.normalization}")

                if self.reduce == 'mean':
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif self.reduce == 'max':
                    field_norm = ops.amax(field_norm, 1)  # [batch, mul]
                else:
                    raise ValueError(f"Invalid reduce option {self.reduce}")

                if not self.instance:
                    field_norm = field_norm.mean(0)  # [mul]
                    new_vars.append(self._roll_avg(self.running_var[irv: irv + mul], field_norm))
            else:
                field_norm = self.running_var[irv: irv + mul]
            irv += mul

            field_norm = (field_norm + self.eps).pow(-0.5)  # [(batch,) mul]

            if self.affine:
                weight = self.weight[iw: iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * weight  # [(batch,) mul]

            field = field * field_norm.reshape(-1, 1, mul, 1)  # [batch, sample, mul, repr]

            if self.affine and ir.is_scalar():  # scalars
                bias = self.bias[ib: ib + mul]  # [mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch, sample, mul, repr]

            fields.append(field.reshape(batch, -1, mul * d))  # [batch, sample, mul * repr]

        if self.training and not self.instance:
            ops.assign(self.running_mean, ops.cat(new_means))
            ops.assign(self.running_var, ops.cat(new_vars))

        output = ops.cat(fields, 2)
        return output.reshape(inputs_shape)
