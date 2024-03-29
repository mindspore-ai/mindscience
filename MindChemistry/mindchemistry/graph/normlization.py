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
"""norm"""
import mindspore as ms
from mindspore import ops, Parameter, nn
from mindchemistry.graph.graph import AggregateNodeToGlobal, LiftGlobalToNode


class BatchNormMask(nn.Cell):
    """BatchNormMask"""

    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.moving_mean = Parameter(ops.zeros((num_features,), ms.float32), name="moving_mean", requires_grad=False)
        self.moving_variance = Parameter(ops.ones((num_features,), ms.float32),
                                         name="moving_variance",
                                         requires_grad=False)
        if affine:
            self.gamma = Parameter(ops.ones((num_features,), ms.float32), name="gamma", requires_grad=True)
            self.beta = Parameter(ops.zeros((num_features,), ms.float32), name="beta", requires_grad=True)

    def construct(self, x, mask, num):
        """construct"""
        if x.shape[1] != self.num_features:
            raise ValueError(f"x.shape[1] {x.shape[1]} is not equal to num_features {self.num_features}")
        if x.shape[0] != mask.shape[0]:
            raise ValueError(f"x.shape[0] {x.shape[0]} is not equal to mask.shape[0] {mask.shape[0]}")

        if x.ndim != mask.ndim:
            if mask.size != mask.shape[0]:
                raise ValueError(f"mask.ndim dose not match src.ndim, and cannot be broadcasted to the same")
            shape = [1] * x.ndim
            shape[0] = -1
            mask = ops.reshape(mask, shape).astype(x.dtype)
        x = ops.mul(x, mask)

        if num == 0:
            raise ValueError

        # pylint: disable=R1705
        if x.ndim > 2:
            norm_axis = []
            shape = [-1]
            for i in range(2, x.ndim):
                norm_axis.append(i)
                shape.append(1)

            if self.training:
                mean = ops.div(ops.sum(x, 0), num)
                mean = ops.mean(mean, norm_axis)
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
                mean = ops.reshape(mean, shape)
                mean = ops.mul(mean, mask)
                x = x - mean

                var = ops.div(ops.sum(ops.pow(x, 2), 0), num)
                var = ops.mean(var, norm_axis)
                self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * var
                std = ops.sqrt(ops.add(var, self.eps))
                std = ops.reshape(std, shape)
                y = ops.true_divide(x, std)
            else:
                mean = ops.reshape(self.moving_mean.astype(x.dtype), shape)
                mean = ops.mul(mean, mask)
                std = ops.sqrt(ops.add(self.moving_variance.astype(x.dtype), self.eps))
                std = ops.reshape(std, shape)
                y = ops.true_divide(ops.sub(x, mean), std)

            if self.affine:
                gamma = ops.reshape(self.gamma.astype(x.dtype), shape)
                beta = ops.reshape(self.beta.astype(x.dtype), shape) * mask
                y = y * gamma + beta

            return y
        else:
            if self.training:
                mean = ops.div(ops.sum(x, 0), num)
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
                mean = ops.mul(mean, mask)
                x = x - mean

                var = ops.div(ops.sum(ops.pow(x, 2), 0), num)
                self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * var
                std = ops.sqrt(ops.add(var, self.eps))
                y = ops.true_divide(x, std)
            else:
                mean = ops.mul(self.moving_mean.astype(x.dtype), mask)
                std = ops.sqrt(ops.add(self.moving_variance.astype(x.dtype), self.eps))
                y = ops.true_divide(ops.sub(x, mean), std)

            if self.affine:
                beta = self.beta.astype(x.dtype) * mask
                y = y * self.gamma.astype(x.dtype) + beta

            return y


class GraphLayerNormMask(nn.Cell):
    """GraphLayerNormMask"""

    def __init__(self,
                 normalized_shape,
                 begin_norm_axis=-1,
                 eps=1e-5,
                 sub_mean=True,
                 divide_std=True,
                 affine_weight=True,
                 affine_bias=True,
                 aggr_mode="mean"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.begin_norm_axis = begin_norm_axis
        self.eps = eps
        self.sub_mean = sub_mean
        self.divide_std = divide_std
        self.affine_weight = affine_weight
        self.affine_bias = affine_bias
        self.mean = ops.ReduceMean(keep_dims=True)
        self.aggregate = AggregateNodeToGlobal(mode=aggr_mode)
        self.lift = LiftGlobalToNode(mode="multi_graph")

        if affine_weight:
            self.gamma = Parameter(ops.ones(normalized_shape, ms.float32), name="gamma", requires_grad=True)
        if affine_bias:
            self.beta = Parameter(ops.zeros(normalized_shape, ms.float32), name="beta", requires_grad=True)

    def construct(self, x, batch, mask, dim_size, scale=None):
        """construct"""
        begin_norm_axis = self.begin_norm_axis if self.begin_norm_axis >= 0 else self.begin_norm_axis + x.ndim
        if begin_norm_axis not in range(1, x.ndim):
            raise ValueError(f"begin_norm_axis {begin_norm_axis} is not in range 1 to {x.ndim}")

        norm_axis = []
        for i in range(begin_norm_axis, x.ndim):
            norm_axis.append(i)
            if self.normalized_shape[i - begin_norm_axis] != x.shape[i]:
                raise ValueError(f"x.shape[{i}] {x.shape[i]} is not equal to normalized_shape[{i - begin_norm_axis}] "
                                 f"{self.normalized_shape[i - begin_norm_axis]}")

        if x.shape[0] != mask.shape[0]:
            raise ValueError(f"x.shape[0] {x.shape[0]} is not equal to mask.shape[0] {mask.shape[0]}")
        if x.shape[0] != batch.shape[0]:
            raise ValueError(f"x.shape[0] {x.shape[0]} is not equal to batch.shape[0] {batch.shape[0]}")

        if x.ndim != mask.ndim:
            if mask.size != mask.shape[0]:
                raise ValueError(f"mask.ndim dose not match src.ndim, and cannot be broadcasted to the same")
            shape = [1] * x.ndim
            shape[0] = -1
            mask = ops.reshape(mask, shape).astype(x.dtype)
        x = ops.mul(x, mask)

        if self.sub_mean:
            mean = self.aggregate(x, batch, dim_size=dim_size, mask=mask)
            mean = self.mean(mean, norm_axis)
            mean = self.lift(mean, batch)
            mean = ops.mul(mean, mask)
            x = x - mean

        if self.divide_std:
            var = self.aggregate(ops.square(x), batch, dim_size=dim_size, mask=mask)
            var = self.mean(var, norm_axis)
            if scale is not None:
                var = var * scale
            std = ops.sqrt(var + self.eps)
            std = self.lift(std, batch)
            x = ops.true_divide(x, std)

        if self.affine_weight:
            x = x * self.gamma.astype(x.dtype)

        if self.affine_bias:
            beta = ops.mul(self.beta.astype(x.dtype), mask)
            x = x + beta

        return x


class GraphInstanceNormMask(nn.Cell):
    """GraphInstanceNormMask"""

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 sub_mean=True,
                 divide_std=True,
                 affine_weight=True,
                 affine_bias=True,
                 aggr_mode="mean"):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.sub_mean = sub_mean
        self.divide_std = divide_std
        self.affine_weight = affine_weight
        self.affine_bias = affine_bias
        self.mean = ops.ReduceMean(keep_dims=True)
        self.aggregate = AggregateNodeToGlobal(mode=aggr_mode)
        self.lift = LiftGlobalToNode(mode="multi_graph")

        if affine_weight:
            self.gamma = Parameter(ops.ones((self.num_features,), ms.float32), name="gamma", requires_grad=True)
        if affine_bias:
            self.beta = Parameter(ops.zeros((self.num_features,), ms.float32), name="beta", requires_grad=True)

    def construct(self, x, batch, mask, dim_size, scale=None):
        """construct"""
        if x.shape[1] != self.num_features:
            raise ValueError(f"x.shape[1] {x.shape[1]} is not equal to num_features {self.num_features}")
        if x.shape[0] != mask.shape[0]:
            raise ValueError(f"x.shape[0] {x.shape[0]} is not equal to mask.shape[0] {mask.shape[0]}")
        if x.shape[0] != batch.shape[0]:
            raise ValueError(f"x.shape[0] {x.shape[0]} is not equal to batch.shape[0] {batch.shape[0]}")

        if x.ndim != mask.ndim:
            if mask.size != mask.shape[0]:
                raise ValueError(f"mask.ndim dose not match src.ndim, and cannot be broadcasted to the same")
            shape = [1] * x.ndim
            shape[0] = -1
            mask = ops.reshape(mask, shape).astype(x.dtype)
        x = ops.mul(x, mask)

        if x.ndim > 2:
            norm_axis = []
            shape = [-1]
            for i in range(2, x.ndim):
                norm_axis.append(i)
                shape.append(1)

            if self.affine_weight:
                gamma = ops.reshape(self.gamma.astype(x.dtype), shape)
            if self.affine_bias:
                beta = ops.reshape(self.beta.astype(x.dtype), shape)
        else:
            if self.affine_weight:
                gamma = self.gamma.astype(x.dtype)
            if self.affine_bias:
                beta = self.beta.astype(x.dtype)

        if self.sub_mean:
            mean = self.aggregate(x, batch, dim_size=dim_size, mask=mask)
            if x.ndim > 2:
                mean = self.mean(mean, norm_axis)
            mean = self.lift(mean, batch)
            mean = ops.mul(mean, mask)
            x = x - mean

        if self.divide_std:
            var = self.aggregate(ops.square(x), batch, dim_size=dim_size, mask=mask)
            if x.ndim > 2:
                var = self.mean(var, norm_axis)
            if scale is not None:
                var = var * scale
            std = ops.sqrt(var + self.eps)
            std = self.lift(std, batch)
            x = ops.true_divide(x, std)

        if self.affine_weight:
            x = x * gamma

        if self.affine_bias:
            beta = ops.mul(beta, mask)
            x = x + beta

        return x
