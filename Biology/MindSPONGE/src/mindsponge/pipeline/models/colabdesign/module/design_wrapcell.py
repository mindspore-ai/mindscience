# Copyright 2023 Huawei Technologies Co., Ltd
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
"""design wrapcell"""
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean, _get_parallel_mode)
from mindspore.context import ParallelMode

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = float(0.001)

clip_grad = ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """clip grad"""
    if clip_type not in (0, 1):
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, ops.cast(ops.tuple_to_array((-clip_value,)), dt),
                                     ops.cast(ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """grad scale"""
    return grad * ops.Reciprocal()(scale)


grad_mul = C.MultitypeFuncGraph("grad_mul")


@grad_mul.register("Tuple", "Tensor")
def tensor_grad_mul(x, y):
    """grad mul"""
    return x * y


grad_square = C.MultitypeFuncGraph("grad_square")


@grad_square.register("Tensor")
def tensor_grad_square(x):
    """grad square"""
    x_temp = ops.Square()(x).astype(mstype.float32)
    x_square = ((x_temp.sum(-1, keepdims=True) > 0).astype(mstype.float32))
    x_square = x_square.sum(-2, keepdims=True).astype(mstype.float32)
    x_sqrt = ops.Sqrt()(x_square).astype(mstype.float32)
    x_final = ops.div(x_sqrt, GRADIENT_CLIP_VALUE)
    return x_final[0][0][0]


class TrainOneStepCell(nn.Cell):
    """TrainOneStepCell"""

    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True, use_global_norm=True):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.enable_clip_grad = enable_clip_grad
        self.hyper_map = ops.HyperMap()
        self.use_global_norm = use_global_norm

        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        """construct"""
        loss = self.network(
            *inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, (
            sens))
        grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_tensor(self.sens)), grads)
        if self.enable_clip_grad:
            if self.use_global_norm:
                eff_len = self.hyper_map(grad_square, grads)
                grads = C.clip_by_global_norm(grads, GRADIENT_CLIP_VALUE)
                grads = self.hyper_map(ops.partial(grad_mul, eff_len), grads)
            else:
                grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)

        loss = F.depend(loss, self.optimizer(grads))
        return loss


class WithLossCell(nn.Cell):
    """WithLossCell"""

    def __init__(self, backbone):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone

    def construct(self, *inputs):
        """construct"""
        out = self._backbone(*inputs)
        return out
