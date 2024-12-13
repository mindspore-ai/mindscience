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
# ==============================================================================
"""the callback functions of SKNO"""
import mindspore.numpy as msnp
from mindspore import nn, ops, Tensor
from mindspore.train.callback import Callback
from mindearth.module import WeatherForecast

def divide_function(numerator, denominator):
    r"""
    Check denominator, and compute the divide
    """
    if isinstance(denominator, (int, float)):
        if denominator != 0:
            result = numerator / denominator
        else:
            raise ValueError("The numerator is divided by Zero!")
    else:
        if denominator.all() != 0:
            result = numerator / denominator
        else:
            raise ValueError("The numerator is divided by Zero!")
    return result


class CustomWithLossCell(nn.Cell):
    r"""
    CustomWithLossCell is used to Connect the feedforward network and multi-label loss function.

    Args:
        backbone: a feedforward neural network
        loss_fn: a multi-label loss function

    Inputs:
        - **data** (Tensor) - The input data of feedforward neural network. Tensor of any dimension.
        - **label** (Tensor) - The input label. Tensor of any dimension.

    Outputs:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        output, recons = self._backbone(data)
        loss = self._loss_fn(output, recons, label, data)
        return loss


class MultiMSELoss(nn.LossBase):
    r"""
    MultiMSELoss is used to calculate multiple MSELoss, then weighted summation. the MSEloss is used to calculate
    the mean squared error between the predicted value and the label value.

    Inputs:
        - **prediction1** (Tensor) - The predicted value of the input. Tensor of any dimension.
        - **prediction2** (Tensor) - The predicted value of the input. Tensor of any dimension.
        - **label1** (Tensor) - The input label. Tensor of any dimension.
        - **label2** (Tensor) - The input label. Tensor of any dimension.
        - **weight1** (Tensor) - The coefficient of l1.
        - **weight2** (Tensor) - The coefficient of l2.

    Outputs:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindearth.loss import MultiMSELoss
        >>> # Case: prediction.shape = labels.shape = (3, 3)
        >>> prediction1 = Tensor(np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> prediction2 = Tensor(np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> label1 = Tensor(np.array([[1, 2, 2],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> label2 = Tensor(np.array([[1, 2, 2],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> loss_fn = MultiMSELoss()
        >>> loss = loss_fn(prediction1, prediction2, label1, label2)
        >>> print(loss)
        0.111111
    """

    def __init__(self, ai, wj, sj_std, feature_dims, use_weight=False):
        super(MultiMSELoss, self).__init__()
        self.loss = nn.MSELoss()
        self.wj = wj
        self.ai = ai
        self.sj_std = sj_std
        self.feature_dims = feature_dims
        self.use_weight = use_weight

    def construct(self, prediction1, prediction2, label1, label2, weight1=0.9, weight2=0.1):
        """Custom loss forward function"""
        prediction1 = prediction1.reshape(-1, self.feature_dims)
        prediction2 = prediction2.reshape(-1, self.feature_dims)
        label1 = label1.reshape(-1, self.feature_dims)
        label2 = label2.reshape(-1, self.feature_dims)

        err1 = msnp.square(prediction1 - label1)
        weighted_err1 = err1 * self.wj * ops.div(self.ai, self.sj_std)
        l1 = msnp.average(weighted_err1)

        err2 = msnp.square(prediction2 - label2)
        weighted_err2 = err2 * self.wj * ops.div(self.ai, self.sj_std)
        l2 = msnp.average(weighted_err2)
        return weight1 * l1 + weight2 * l2


class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.
    """

    def __init__(self, model, config, logger):
        super().__init__(model, config, logger)
        self.model = model
        self.config = config
        self.logger = logger

    def forecast(self, inputs):
        pred_lst = []
        for _ in range(self.t_out):
            pred, _ = self.model(inputs)
            pred_lst.append(pred.transpose(0, 2, 3, 1).reshape(self.batch_size, self.h_size * self.w_size,
                                                               self.feature_dims).asnumpy())
            inputs = pred
        return pred_lst



class EvaluateCallBack(Callback):
    """
    Monitor the prediction accuracy in training.
    """

    def __init__(self,
                 model,
                 valid_dataset_generator,
                 config,
                 logger
                 ):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.valid_dataset_generator = valid_dataset_generator
        self.predict_interval = config['summary']["valid_frequency"]
        self.logger = logger
        self.eval_net = InferenceModule(model, config, logger)

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            self.eval_net.eval(self.valid_dataset_generator, generator_flag=True)


class Lploss(nn.LossBase):
    r"""
    The loss function uses the normalized difference as the loss function value of the model.

    .. math::

        Lploss = \frac{ \sqrt[p]{(\sum (abs(Y_{pred}-Y_{label})))^p} }{ \sqrt[p]{(\sum(Y_{label}))^p} }, (p=2),

    ...

    Inputs:
        - **prediction1** (Tensor) - The predicted value of the input. Tensor of any dimension.
        - **prediction2** (Tensor) - The predicted value of the input. Tensor of any dimension.
        - **label1** (Tensor) - The input label. Tensor of any dimension.
        - **label2** (Tensor) - The input label. Tensor of any dimension.
        - **weight1** (Tensor) - The coefficient of l1.
        - **weight2** (Tensor) - The coefficient of l2.

    Outputs:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindearth.loss import MultiMSELoss
        >>> # Case: prediction.shape = labels.shape = (3, 3)
        >>> prediction1 = Tensor(np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> prediction2 = Tensor(np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> label1 = Tensor(np.array([[1, 2, 2],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> label2 = Tensor(np.array([[1, 2, 2],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
        >>> loss_fn = Lploss()
        >>> loss = loss_fn(prediction1, prediction2, label1, label2)
        >>> print(loss)
        0.111111
    """

    def __init__(self, p=2, size_average=True, reduction=True):
        super(Lploss, self).__init__()
        # Dimension and Lp-norm type are positive
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def loss(self, x, y):
        """loss"""
        num_examples = x.shape[0]

        diff_norms = ops.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = ops.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                loss = ops.mean(ops.div(diff_norms, y_norms))
            else:
                loss = Tensor.sum(ops.div(diff_norms, y_norms))
        else:
            loss = ops.div(diff_norms, y_norms)
        return loss

    def construct(self, prediction1, prediction2, label1, label2, weight1=0.8, weight2=0.2):
        l1 = self.loss(prediction1, label1)
        l2 = self.loss(prediction2, label2)
        return weight1 * l1 + weight2 * l2
