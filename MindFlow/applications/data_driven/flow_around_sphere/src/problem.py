# ============================================================================
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
"""self-defined problem: 3D Unsteady Flow"""
from mindspore import nn, ops, jit_class

from mindflow.loss import get_loss_metric
from mindflow.pde import UnsteadyFlowWithLoss
from mindflow.utils.check_func import check_param_type

from src import RRMSE, GradientRRMSE


@jit_class
class UnsteadyFlow3D(UnsteadyFlowWithLoss):
    """
    Base class of unsteady self-defined data-driven problems.

    Args:
        model(mindspore.nn.Cell): A training or test model.
        loss_fn(Union[str, Cell]): Loss function. Default: ``"mse"``.
        t_in(int): Initial time steps. Default: ``1``.
        t_out(int): Output time steps. Default: ``1``.
        residual(bool): Whether to predict the amount of change. Default: ``True``.
        scale(float): Magnification of target flow field only valid when residual=True. Default: ``1000.0``.

    Supported Platforms:
        ``Ascend`` ``GPU``
    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_channel=4, out_channel=4):
        ...         super().__init__()
        ...         self.conv1 = nn.Conv2d(in_channel, 8, 3, pad_mode='same')
        ...         self.conv2 = nn.Conv2d(8, 16, 3, pad_mode='same')
        ...         self.relu = nn.ReLU()
        ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        ...         self.deconv1 = nn.Conv2dTranspose(16, 8, 3, stride=2)
        ...         self.deconv2 = nn.Conv2dTranspose(8, out_channel, 3, stride=2)
        ...     def construct(self, x):
        ...         x = self.max_pool2d(self.relu(self.conv1(x)))
        ...         x = self.max_pool2d(self.relu(self.conv2(x)))
        ...         x = self.deconv1(x)
        ...         x = self.deconv2(x)
        ...         return x
        >>> model = Net()
        >>> problem = UnsteadyFlow3D(model=model)
        >>> # Case: inputs.shape = (bs, t_in, C, H, W)
        >>> # Case: prediction.shape = labels.shape = (bs, t_out, C, H, W)
        >>> inputs = Tensor(np.random.randn(32, 1, 4, 32, 32), mindspore.float32)
        >>> pred, _ = problem.step(inputs)
        >>> print(pred.shape)
        (32, 1, 4, 32, 32)
        >>> label = Tensor(np.random.randn(32, 1, 4, 32, 32), mindspore.float32)
        >>> loss = problem.get_loss(inputs, label)
        >>> print(loss)
        0.99785775
        >>> metric = problem.get_metric(inputs, label)
        >>> print(metric)
        0.7975668
    """

    def __init__(self, model, loss_fn='mse', metric_fn='mae', loss_weight=100.0, dynamic_flag=True, t_in=1,
                 t_out=1, residual=True, scale=1000.0):
        if loss_fn == 'RRMSE':
            loss_fn = RRMSE()
        elif loss_fn == 'GradientRRMSE':
            loss_fn = GradientRRMSE(loss_weight=loss_weight, dynamic_flag=dynamic_flag)
        else:
            loss_fn = get_loss_metric(loss_fn)
        super().__init__(model, t_in, t_out, loss_fn)

        self.scale = scale
        self.residual = residual

        if metric_fn == 'RRMSE':
            self.metric_fn = RRMSE()
        else:
            self.metric_fn = get_loss_metric(metric_fn)

        check_param_type(self.model, "model", data_type=nn.Cell, exclude_type=bool)
        check_param_type(self.loss_fn, "loss_fn", data_type=nn.Cell, exclude_type=bool)
        check_param_type(self.metric_fn, "metric_fn", data_type=nn.Cell, exclude_type=bool)
        check_param_type(self.t_in, "t_in", data_type=int, exclude_type=bool)
        check_param_type(self.t_out, "t_out", data_type=int, exclude_type=bool)

    def step(self, inputs):
        """
        Support single or multiple time steps training.

        Args:
            inputs (Tensor): Input dataset with data format is "NTCDHW" .

        Returns:
            List(Tensor), Dataset with data format is "NTCDHW".
        """
        pred_list = []
        for _ in range(self.t_out):
            # change inputs dimension: inputs[bs, t_in, c, x1, x2, ...] -> inp[b, t_in * c, x1, x2, ...]
            inp = self._flatten(inputs)
            pred = self.model(inp)  # -> [b, c, x1, x2, ...]
            pred = pred.expand_dims(axis=1)  # -> [bs, 1, c, x1, x2, ...]
            pred_list.append(pred)
            inputs = ops.concat([inputs[:, 1:, ...], self.residual * inputs[:, 0:1, ...] + pred / self.scale], axis=1)

        pred_list = ops.concat(pred_list, axis=1)  # -> [bs, t_out, c, x1, x2, ...]
        updatad_inputs = inputs
        return pred_list, updatad_inputs

    def get_loss(self, inputs, target):
        """
        Compute the loss during training progress.

        Args:
            inputs(Tensor): Dataset with data format is "NTCDHW".
            target(Tensor): True values of the samples. "NTCDHW"

        Returns:
            float, loss value.
        """
        # the dimension of inputs: [bs, t_in, c, x1, x2, ...]
        # the dimension of labels [bs, t_out, c, x1, x2, ...]
        pred, _ = self.step(inputs)
        loss = self.loss_fn(pred, target)
        return loss

    def get_metric(self, inputs, target):
        """
        Compute the uniform metric during eval and infer progress.

        Args:
            inputs(Tensor): Dataset with data format is "NTCDHW".
            target(Tensor): True values of the samples. "NTCDHW".

        Returns:
            float, metric value.
        """
        pred, _ = self.step(inputs)
        metric = self.metric_fn(pred, target)
        return metric.asnumpy()
