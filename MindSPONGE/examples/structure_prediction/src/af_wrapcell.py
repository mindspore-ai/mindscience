# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""warp cell"""

import mindspore.nn as nn
from mindspore import ops
from mindspore.context import ParallelMode
from mindspore.nn import DistributedGradReducer
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_device_num
from mindspore.parallel._utils import (_get_gradients_mean, _get_parallel_mode)

GRADIENT_CLIP_TYPE = 1

clip_grad = ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """_clip_grad"""
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
    """tensor_grad_scale"""
    return grad * ops.Reciprocal()(scale)


class TrainOneStepCell(nn.Cell):
    """TrainOneStepCell"""

    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True, use_global_norm=True,
                 gradient_clip_value=1.0):
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
        self.gradient_clip_value = gradient_clip_value

        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        """construct"""
        if self.train_backward:
            loss = self.network(*inputs)
            loss, l_fape_side, l_fape_backbone, l_anglenorm, distogram_loss, masked_loss, predict_lddt_loss = loss
            sens = F.fill(loss.dtype, loss.shape, self.sens)
            sens1 = F.fill(l_fape_side.dtype, l_fape_side.shape, 0.0)
            sens2 = F.fill(l_fape_backbone.dtype, l_fape_backbone.shape, 0.0)
            sens3 = F.fill(l_anglenorm.dtype, l_anglenorm.shape, 0.0)
            sens4 = F.fill(distogram_loss.dtype, distogram_loss.shape, 0.0)
            sens5 = F.fill(masked_loss.dtype, masked_loss.shape, 0.0)
            sens6 = F.fill(predict_lddt_loss.dtype, predict_lddt_loss.shape, 0.0)

            grads = self.grad(self.network, self.weights)(*inputs, (sens, sens1, sens2, sens3, sens4, sens5, sens6))
            grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_array(self.sens)), grads)
            grads = self.grad_reducer(grads)
            if self.enable_clip_grad:
                if self.use_global_norm:
                    grads = C.clip_by_global_norm(grads, self.gradient_clip_value)
                else:
                    grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, self.gradient_clip_value), grads)

            loss = F.depend(loss, self.optimizer(grads))
            return loss, l_fape_side, l_fape_backbone, l_anglenorm, distogram_loss, masked_loss, predict_lddt_loss

        out = self.network(*inputs)
        return out


class WithLossCell(nn.Cell):
    """WithLossCell"""

    def __init__(self, backbone):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone

    def construct(self, *inputs):
        """construct"""
        out = self._backbone(*inputs)
        return out
