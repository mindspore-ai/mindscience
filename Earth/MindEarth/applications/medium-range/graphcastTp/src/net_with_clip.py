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
# ==============================================================================
"""grad clip"""
from mindspore import ops, nn
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.nn import TrainOneStepWithLossScaleCell

gradient_clip_type = 1
clip_grad = ops.MultitypeFuncGraph("clip_grad")
grad_scale = C.MultitypeFuncGraph("grad_scale")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """_clip_grad"""
    if clip_type not in (0, 1):
        return grad
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, ops.cast(ops.tuple_to_array((-clip_value,)), grad.dtype),
                                     ops.cast(ops.tuple_to_array((clip_value,)), grad.dtype))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), grad.dtype))
    return new_grad


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """tensor_grad_scale"""
    return grad * ops.Reciprocal()(scale)


class TrainOneStepCell(TrainOneStepWithLossScaleCell):
    """
    Network training with loss scaling.
    """
    def __init__(self,
                 network,
                 optimizer,
                 scale_sense,
                 enable_clip_grad=False,
                 use_global_norm=True,
                 gradient_clip_value=32):
        super(TrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.use_global_norm = use_global_norm
        self.gradient_clip_value = gradient_clip_value
        self.enable_clip_grad = enable_clip_grad

    def construct(self, *inputs):
        """gradient clip"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            if self.enable_clip_grad:
                if self.use_global_norm:
                    grads = C.clip_by_global_norm(grads, self.gradient_clip_value)
                else:
                    grads = self.hyper_map(ops.partial(clip_grad, gradient_clip_type, self.gradient_clip_value), grads)
            loss = F.depend(loss, self.optimizer(grads))
        return loss
