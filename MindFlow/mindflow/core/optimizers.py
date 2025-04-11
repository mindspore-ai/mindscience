# ============================================================================
# Copyright 2025 Huawei Technologies Co., Ltd
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
''' Define costumized optimizers and gradient accumulators '''
import mindspore as ms
from mindspore import nn, ops


class AdaHessian(nn.Adam):
    r"""
    The Adahessian optimizer.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning
    <https://arxiv.org/abs/2006.00719>`_ .
    See the `Torch implementation
    <https://github.com/amirgholami/adahessian/blob/master/instruction/adahessian.py>`_  for reference.
    The Hessian power here is fixed to 1, and the way of spatially averaging the Hessian traces follows the default
    behavior in the Torch implementation, that is

    - for 1D: no spatial average.
    - for 2D: use the entire row as the spatial average.
    - for 3D (assume 1D Conv, can be customized): use the last dimension as spatial average.
    - for 4D (assume 2D Conv, can be customized): use the last 2 dimensions as spatial average.

    Args see `mindspore.nn.Adam <https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Adam.html>`_ .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import ops, nn
        >>> from mindflow import AdaHessian
        >>> ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
        >>> net = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3)
        >>> def forward(a):
        >>>     return ops.mean(net(a)**2)**.5
        >>> grad_fn = ms.grad(forward, grad_position=None, weights=net.trainable_params())
        >>> optimizer = AdaHessian(net.trainable_params())
        >>> inputs = ms.Tensor(np.reshape(range(100), [2, 2, 5, 5]), dtype=ms.float32)
        >>> optimizer(grad_fn, inputs)
        >>> print(optimizer.moment2[0].shape)
        (4, 2, 3, 3)
    """

    def gen_rand_vecs(self, grads):
        return [(2 * ops.randint(0, 2, p.shape) - 1).astype(ms.float32) for p in grads]

    def _modify_moments(self, grad_fn, inputs):
        """ introduce Hutchinson trace by pre-adding its difference to grads' square into the second moment
        """
        # generate the function for 2nd-order derivative
        # vjp_fn solve for the derivative of both input and weights
        grads, vjp_fn = ms.vjp(grad_fn, inputs, weights=self.parameters)

        # generate random vector
        vs = self.gen_rand_vecs(grads)

        # solve for hutchinson trace
        # when operator does not support 2nd-order derivative by vjp(), using `hvs = grads` instead
        # to make the code run, but the output value would not be correct
        _, hvs = vjp_fn(tuple(vs))

        hutchinson_trace = []

        for hv in hvs:
            hv_abs = hv.abs()

            if hv.ndim <= 1:
                hutchinson_trace.append(hv_abs)
            elif hv.ndim == 2:
                hutchinson_trace.append(ops.mean(hv_abs, axis=[1], keep_dims=True))
            elif hv.ndim == 3:
                hutchinson_trace.append(ops.mean(hv_abs, axis=[2], keep_dims=True))
            elif hv.ndim == 4:
                hutchinson_trace.append(ops.mean(hv_abs, axis=[2, 3], keep_dims=True))
            else:
                raise RuntimeError(f'You need to write your customized function to support this shape: {hv.shape}')

        # modify moment2
        for i in range(len(self.moment2)):
            ops.assign(
                self.moment2[i],
                self.moment2[i] + (1. - self.beta2) * (
                    hutchinson_trace[i] + grads[i]) * (hutchinson_trace[i] - grads[i]) / self.beta2)

        return grads

    def construct(self, grad_fn, inputs):
        """Update the weights using AdaHessian algorithm
        Args:
            grad_fn (callable): the function that outputs 1st-order gradients
            inputs (Tensor): the inputs to the gradient function
        """
        gradients = self._modify_moments(grad_fn, inputs)

        params = self._parameters
        moment1 = self.moment1
        moment2 = self.moment2
        gradients = self.flatten_gradients(gradients)
        gradients = self.decay_weight(gradients)
        if not self.use_offload:
            gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        gradients = self._grad_sparse_indices_deduplicate(gradients)
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        self.beta1_power *= self.beta1
        self.beta2_power *= self.beta2

        return self._apply_adam(params, self.beta1_power, self.beta2_power, moment1, moment2, lr, gradients)
