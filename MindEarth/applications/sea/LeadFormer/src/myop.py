# right 2020-2022 Huawei Technologies Co., Ltd
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
"""adam"""
from __future__ import absolute_import, division
import numpy as np

from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor
from mindspore import _checkparam as validator
from mindspore.nn.optim.optimizer import Optimizer


_adam_opt = C.MultitypeFuncGraph("adam_opt")
_fused_adam_weight_decay = C.MultitypeFuncGraph("fused_adam_weight_decay")
_lazy_adam_opt = C.MultitypeFuncGraph("lazy_adam_opt")
_scaler_one = Tensor(1, mstype.int32)
_scaler_ten = Tensor(10, mstype.float32)


@_adam_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_op(beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flag, optim_filter):
    """Update parameters."""
    op_cast = P.Cast()
    if optim_filter:
        op_mul = P.Mul()
        op_square = P.Square()
        op_sqrt = P.Sqrt()
        op_cast = P.Cast()
        op_reshape = P.Reshape()
        op_shape = P.Shape()
        param_fp32 = op_cast(param, mstype.float32)
        m_fp32 = op_cast(m, mstype.float32)
        v_fp32 = op_cast(v, mstype.float32)
        gradient_fp32 = op_cast(gradient, mstype.float32)

        next_m = op_mul(beta1, m_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                                - beta1, gradient_fp32)

        next_v = op_mul(beta2, v_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                                - beta2, op_square(gradient_fp32))

        update = next_m / (eps + op_sqrt(next_v))
        if decay_flag:
            update = op_mul(weight_decay, param_fp32) + update

        update_with_lr = op_mul(lr, update)
        next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))

        next_param = F.depend(next_param, F.assign(param, op_cast(next_param, F.dtype(param))))
        next_param = F.depend(next_param, F.assign(m, op_cast(next_m, F.dtype(m))))
        next_param = F.depend(next_param, F.assign(v, op_cast(next_v, F.dtype(v))))

        return op_cast(next_param, F.dtype(param))
    return op_cast(gradient, F.dtype(param))

def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, validator.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, validator.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)


class AdamWeightDecay(Optimizer):
    """
    Adam optimizer with weight decay.

    This implementation adds weight decay to the standard Adam algorithm.
    It supports parallel optimization and can run on different device targets.
    Args:
            params: Network parameters to optimize
            learning_rate: Learning rate (default: 1e-3)
            beta1: Exponential decay rate for first moment estimates (default: 0.9)
            beta2: Exponential decay rate for second moment estimates (default: 0.999)
            eps: Term added to denominator for numerical stability (default: 1e-6)
            weight_decay: Weight decay coefficient (default: 0.0)
    """
    _support_parallel_optimizer = True

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super().__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self._parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self._parameters.clone(prefix="adam_v", init='zeros')
        self.fused_opt = P.AdamWeightDecay()
        if context.get_context("device_target") == "Ascend":
            self.use_fused_opt = False
        else:
            self.use_fused_opt = True

    @jit
    def construct(self, gradients):
        """construct"""
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        if self.use_fused_opt:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(
                        F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps),
                        lr, weight_decay, self._parameters, self.moments1,
                        self.moments2, gradients, self.decay_flags, self.optim_filter)
                else:
                    optim_result = self.hyper_map(
                        F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps, lr),
                        weight_decay, self._parameters, self.moments1, self.moments2,
                        gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(
                    F.partial(_fused_adam_weight_decay, self.fused_opt, self.beta1, self.beta2, self.eps, lr,
                              weight_decay),
                    self._parameters, self.moments1, self.moments2,
                    gradients, self.decay_flags, self.optim_filter)
        else:
            if self.is_group:
                if self.is_group_lr:
                    optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps),
                                                  lr, weight_decay, self._parameters, self.moments1,
                                                  self.moments2, gradients, self.decay_flags, self.optim_filter)
                else:
                    optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr),
                                                  weight_decay, self._parameters, self.moments1, self.moments2,
                                                  gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr, weight_decay),
                                              self._parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        self._set_base_target(value)
        if value == 'CPU':
            self.fused_opt.set_device("CPU")
            self.use_fused_opt = True
        else:
            self.use_fused_opt = False
