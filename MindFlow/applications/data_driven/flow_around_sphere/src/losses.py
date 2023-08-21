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
"""add self-defined loss function"""
from mindspore import nn, ops
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindflow.loss import MTLWeightedLoss


class RRMSE(nn.LossBase):
    r"""
    RRMSE is the sum of relative root mean squared error of different channels.

    For simplicity, let :math:`X` be 2-dimensional Tensor with shape :math:`(C, N)`, and :math:`x` which is one of
    the channels of :math:`X` be 1-dimensional Tensor with Length :math:`N`, the loss of :math:`X` and :math:` \hat{
    X}` is given as:

    .. math: RRMSE(x, \hat{x})=\frac{\sqrt{\frac{1}{n} \sum_{i=1}^n\left(x_i-\hat{x}_i\right)^2}}{\sqrt{\frac{1}{n}
    \sum_{i=1}^n\left(x_i-\bar{x}\right)^2}} RRMSE(X, \hat{X})=RRMSE(x, \hat{x})+\cdots +RRMSE(y, \hat{y})

    where  :math:`x` is the labels, :math:`\hat{x}` is the prediction, and :math:`\bar{x}` is the mean value of the
    labels.

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are ``"mean"``,
            ``"sum"``, and ``"none"``. Default: ``"mean"``.

    Inputs:
      - **logits** (Tensor) - The prediction value of the network. Tensor of shape :math:`(N, T, C,*)` where :math:`*`
                                means, any number of additional dimensions.
      - **labels** (Tensor) - True value of the samples. Tensor of shape :math:`(N, T, C, *)`,  where :math:`*` means,
                                 any number of additional dimensions, same shape as the `prediction` in common cases.

    Outputs:
        Tensor, weighted loss.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> # Case: prediction.shape = labels.shape = (N, T, C, H, W)
        >>> np.random.seed(123456)
        >>> prediction = Tensor(np.random.randn(10, 1, 4, 128, 64), mindspore.float32)
        >>> labels = Tensor(np.random.randn(10, 1, 4, 128, 64), mindspore.float32)
        >>> loss_fn = RRMSE()
        >>> loss = loss_fn(prediction, labels)
        >>> print(loss)
        1.4133672
    """

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def construct(self, logits, labels):
        """get regularized loss"""
        logits = P.Cast()(logits, mstype.float32)
        logits = logits.swapaxes(1, 2)
        batch_size = logits.shape[0]
        channel_size = logits.shape[2]
        diff_norms = F.square(
            logits.reshape(batch_size, channel_size, -1) - labels.reshape(batch_size, channel_size, -1)).sum(axis=2)
        mean_diff_norms = diff_norms / (logits.reshape(batch_size, channel_size, -1).shape[2])
        label_norms = F.square(
            labels.reshape(batch_size, channel_size, -1) - F.mean(labels.reshape(batch_size, channel_size, -1), axis=2,
                                                                  keep_dims=True)).sum(axis=2)
        mean_label_norms = label_norms / (logits.reshape(batch_size, channel_size, -1).shape[2] - 1)
        rel_error = ops.div(F.sqrt(mean_diff_norms), F.sqrt(mean_label_norms)).sum(axis=1)
        return self.get_loss(rel_error)


def derivation(data, delta=2):
    """
    Calculate the gradient in different directions.

    Args: data(Tensor): Predicted or actual snapshot. Shape: [N, T, C, D, H, W]. delta(int): Gradient distance. When
    delta is 2, the calculation method is the central difference scheme. Default: ``2``.

    Returns:
        Tensor, Derivative values in different directions.
    """
    grad_x = data[:, :, :, :, :, delta:] - data[:, :, :, :, :, :-delta]
    grad_y = data[:, :, :, :, delta:, :] - data[:, :, :, :, :-delta, :]
    grad_z = data[:, :, :, delta:, :, :] - data[:, :, :, :-delta, :, :]
    return grad_x, grad_y, grad_z


class GradientRRMSE(nn.LossBase):
    """
    GradientRRMSE is the relative root mean squared error including gradient loss, which is also calculated for
    different channels.

    Args:
        loss_weight(float): Weight of gradient loss. Default: ``1``.
        dynamic_flag(bool): Flag to use MTLWeightedLossCell. Default: ``True``.

    Inputs:
    - **logits** (Tensor) - The prediction value of the network. Tensor of shape :math:`(N, T, C,  D, H,
    W)` where :math:`*` means, any number of additional dimensions.
    - **labels** (Tensor) - True value of the
    samples. Tensor of shape :math:`(N, T, C, D, H, W)`,  where :math:`*` means, any number of additional dimensions,
    same shape as the `prediction` in common cases.


    Outputs:
        Tensor, weighted loss.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> # Case: prediction.shape = labels.shape = (N, T, C, D, H, W)
        >>> np.random.seed(123456)
        >>> prediction = Tensor(np.random.randn(10, 1, 4, 64, 128, 64), mindspore.float32)
        >>> labels = Tensor(np.random.randn(10, 1, 4, 64, 128, 64), mindspore.float32)
        >>> loss_fn = GradientRRMSE()
        >>> loss = loss_fn(prediction, labels)
        >>> print(loss)
        4.214711
    """

    def __init__(self, loss_weight=100.0, dynamic_flag=True):
        super().__init__()
        self.alpha = loss_weight
        self.dynamic_flag = dynamic_flag
        self.mtl = MTLWeightedLoss(num_losses=2)

    @staticmethod
    def gradient_loss(logits, labels):
        """
        calculate the grad_loss in different channels.
        logits:[bs, T, C, D, H, W]
        labels:[bs, T, C, D, H, W]
        """
        drec_dx, drec_dy, drec_dz = derivation(logits)
        dimgs_dx, dimgs_dy, dimgs_dz = derivation(labels)
        loss_x = RRMSE()(drec_dx, dimgs_dx)
        loss_y = RRMSE()(drec_dy, dimgs_dy)
        loss_z = RRMSE()(drec_dz, dimgs_dz)
        return loss_x + loss_y + loss_z

    def construct(self, logits, labels):
        """get gradient enhanced loss"""
        int_loss = RRMSE()(logits, labels)
        grad_loss = self.gradient_loss(logits, labels)
        if self.dynamic_flag:
            return self.mtl((int_loss, grad_loss))
        return int_loss + self.alpha * grad_loss
