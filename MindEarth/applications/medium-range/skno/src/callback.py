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
import os

import numpy as np
import mindspore as ms
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
        super(InferenceModule, self).__init__(model, config, logger)
        statistic_dir = os.path.join(config["data"]["root_dir"], "statistic")
        mean = np.load(os.path.join(statistic_dir, "mean.npy"))
        mean = mean.transpose(1, 2, 3, 0)  # HWFL(1, 1, 5, 13)
        mean = mean.reshape((1, -1))
        mean = np.squeeze(mean, axis=0)
        mean_s = np.load(os.path.join(statistic_dir, "mean_s.npy"))
        self.mean_all = np.concatenate([mean, mean_s], axis=-1)

        std = np.load(os.path.join(statistic_dir, "std.npy"))
        std = std.transpose(1, 2, 3, 0)
        std = std.reshape((1, -1))
        self.std = np.squeeze(std, axis=0)
        self.std_s = np.load(os.path.join(statistic_dir, "std_s.npy"))
        self.std_all = np.concatenate([self.std, self.std_s], axis=-1)

        self.feature_dims = config['data']['feature_dims']
        self.climate = np.load(os.path.join(statistic_dir, "climate_1.4.npy"))

    def forecast(self, inputs):
        pred_lst = []
        for _ in range(self.t_out_test):
            pred, _ = self.model(inputs)
            pred_lst.append(pred)
            inputs = pred
        return pred_lst

    def _get_metrics(self, inputs, labels):
        """Get lat_weight_rmse and lat_weight_acc metrics"""
        batch_size = inputs.shape[0]
        pred = self.forecast(inputs)
        pred = ops.stack(pred, 1).transpose(0, 1, 3, 4, 2)  # (B,T,C,H W)->BTHWC
        labels = labels.transpose(0, 2, 3, 4, 1)  # (B,C,T,H W)->BTHWC

        pred = pred.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims)
        labels = labels.reshape(batch_size, self.t_out_test, self.h_size * self.w_size, self.feature_dims)

        # rmse
        error = ops.square(pred - labels).transpose(0, 1, 3, 2).reshape(
            batch_size, self.t_out_test * self.feature_dims, -1)
        weight = ms.Tensor(self._calculate_lat_weight().reshape(-1, 1))
        lat_weight_rmse_step = ops.matmul(error, weight).sum(axis=0)
        lat_weight_rmse_step = lat_weight_rmse_step.reshape(self.t_out_test,
                                                            self.feature_dims).transpose(1, 0).asnumpy()

        # acc
        pred = pred * ms.Tensor(self.std_all, ms.float32) + ms.Tensor(self.mean_all, ms.float32)
        labels = labels * ms.Tensor(self.std_all, ms.float32) + ms.Tensor(self.mean_all, ms.float32)
        pred = pred - ms.Tensor(self.climate, ms.float32)
        labels = labels - ms.Tensor(self.climate, ms.float32)

        acc_numerator = pred * labels
        acc_numerator = acc_numerator.transpose(0, 1, 3, 2).reshape(
            batch_size, self.t_out_test * self.feature_dims, -1)
        acc_numerator = ops.matmul(acc_numerator, weight)

        pred_square = ops.square(pred).transpose(0, 1, 3, 2).reshape(
            batch_size, self.t_out_test * self.feature_dims, -1)
        label_square = ops.square(labels).transpose(0, 1, 3, 2).reshape(
            batch_size, self.t_out_test * self.feature_dims, -1)

        acc_denominator = ops.sqrt(ops.matmul(pred_square, weight) * ops.matmul(label_square, weight))
        lat_weight_acc = np.divide(acc_numerator.asnumpy(), acc_denominator.asnumpy())
        lat_weight_acc_step = lat_weight_acc.sum(axis=0).reshape(self.t_out_test,
                                                                 self.feature_dims).transpose(1, 0)
        return lat_weight_rmse_step, lat_weight_acc_step

    def _calculate_lat_weight(self):
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(3.1416 / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        grid_lat_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1)
        return grid_lat_weight.astype(np.float32)


class EvaluateCallBack(Callback):
    """
    Monitor the prediction accuracy in training.
    """

    def __init__(self,
                 model,
                 test_dataset,
                 config,
                 logger
                 ):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.test_dataset = test_dataset
        self.predict_interval = config['summary']["valid_frequency"]
        self.logger = logger
        self.eval_net = InferenceModule(model,
                                        config,
                                        logger)

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            self.eval_net.eval(self.test_dataset)


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
