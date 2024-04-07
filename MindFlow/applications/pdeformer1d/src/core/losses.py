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
"""Loss functions."""
import numpy as np
from mindspore import nn, ops, Tensor
from mindspore import context
from mindspore.ops import functional as F


class PerSampleLossBase(nn.LossBase):
    r"""
    Base class for per-sample losses.

    This class serves as a base for losses that are calculated on a per-sample
    basis. It provides methods to handle the reduction of loss tensors over the
    specified axes, starting from axis 1 to avoid reducing over the sample axis
    (axis 0).

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(M, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases.  However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor, the shape is :math:`(N)`.
    """

    def get_axis(self, x: Tensor):
        shape = F.shape(x)
        length = F.tuple_len(shape)
        # The only difference compared with nn.LossBase: The axis starts from 1
        # instead of 0, thereby avoid reduction over the samples (axis 0).
        perm = F.make_range(1, length)
        return perm


class PerSampleMSELoss(PerSampleLossBase):
    r"""
    Calculates the mean squared error between the predicted value and the label
    value for each sample.

    Args:
        reduction (str, optional): Apply specific reduction method to the
            output for each data sample: ``'none'`` , ``'mean'`` , ``'sum'`` .
            Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the mean of elements in the output.
            - ``'sum'``: the output elements will be summed.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Raises:
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'`` or ``'sum'``.
        ValueError: If `logits` and `labels` have different shapes and cannot be broadcasted.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleMSELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        x = F.square(logits - labels)
        return self.get_loss(x)


class PerSampleRMSELoss(PerSampleLossBase):
    r"""
    PerSampleRMSELoss creates a criterion to measure the root mean square error
    between :math:`x` and :math:`y` for each data sample, where :math:`x` is
    the predicted value and :math:`y` is the label.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleRMSELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def __init__(self) -> None:
        """Initialize PerSampleRMSELoss."""
        super().__init__()
        self.persample_mse_loss = PerSampleMSELoss()

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        rmse_loss = F.sqrt(self.persample_mse_loss(logits, labels))
        return rmse_loss


class PerSampleMAELoss(PerSampleLossBase):
    r"""
    PerSampleMAELoss creates a criterion to measure the average absolute error
    between :math:`x` and :math:`y` for each sample, where :math:`x` is the
    predicted value and :math:`y` is the label.

    Args:
        reduction (str, optional): Apply specific reduction method to the
            output for each data sample: ``'none'`` , ``'mean'`` , ``'sum'`` .
            Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the mean of elements in the output.
            - ``'sum'``: the output elements will be summed.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Raises:
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'`` or ``'sum'``.
        ValueError: If `logits` and `labels` have different shapes and cannot be broadcasted.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleMAELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def construct(self, logits, label):
        x = F.abs(logits - label)
        return self.get_loss(x)


class LossFunction(nn.Cell):
    r"""
    Computes the loss based on the specified loss type and parameters.
    This class encapsulates different types of loss functions and allows for various configurations
    such as normalization, reduction to mean, and causality weighting.

    Args:
        loss_type (str): The type of loss to compute. Supported values are 'MSE', 'RMSE', 'MAE'.
        normalize (bool): Whether to normalize the loss. Default: False.
        reduce_mean (bool): Whether to reduce the loss to mean over the batch. Default: True.
        normalize_eps (float): A small value to add to the denominator to avoid division by zero
            during normalization. Default: 0.01.
    """

    def __init__(self,
                 loss_type: str,
                 normalize: bool = False,
                 reduce_mean: bool = True,
                 normalize_eps: float = 0.01) -> None:
        super().__init__()
        self.normalize = normalize
        self.reduce_mean = reduce_mean
        self.normalize_eps = normalize_eps
        if self.normalize_eps <= 0:
            raise ValueError(
                f"'normalize_eps' should be a positive float, but got '{normalize_eps}'.")

        # sample_loss_fn
        if loss_type == "MSE":
            self.sample_loss_fn = PerSampleMSELoss()
        elif loss_type == "RMSE":
            self.sample_loss_fn = PerSampleRMSELoss()
        elif loss_type == "MAE":
            self.sample_loss_fn = PerSampleMAELoss()
        else:
            raise ValueError(
                "'loss_type' should be one of ['MSE', 'RMSE', 'MAE'], "
                f"but got '{loss_type}'.")

    def construct(self, pred: Tensor, label: Tensor) -> Tensor:
        loss = self.sample_loss_fn(pred, label)  # shape [bsz]
        if self.normalize:
            label_norm = self.sample_loss_fn(label, 0 * label)  # [bsz]
            loss = loss / (label_norm + self.normalize_eps)  # [bsz]
        if self.reduce_mean:
            loss = ops.mean(loss)  # shape []
        return loss


if __name__ == "__main__":  # unit test
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    loss_fn = PerSampleMAELoss()
    x1 = Tensor(np.random.rand(3, 2, 2))
    x2 = Tensor(np.random.rand(3, 2, 2))
    raw_loss = loss_fn(x1, x2)
    print(raw_loss)
