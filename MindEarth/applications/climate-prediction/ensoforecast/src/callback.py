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
"""eval and loss"""
import numpy as np
import scipy.stats as sps

import mindspore as ms
from mindspore import Tensor, ops, nn
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord


class WeightedLoss(nn.LossBase):
    r"""
    This class is used to define the weighted loss function of CHEFNet. The loss function consists of three items,
    namely MSE in the prediction period, MSE in the observation period, and correlation coefficient loss.

    Args:
        lambda_1 (float): The weight of MSE in the prediction period.Default: ``1.``.
        lambda_2 (float): The weight of MSE in the observation period.Default: ``0.02``.
        lambda_3 (float): The weight of correlation coefficient loss.Default: ``0.6``.
        corr_point (float): Hyperparameters for correlation coefficient loss.Default: ``0.5``.
        obs_time (int): The length of time the data can be observed.Default: ``12``.

    Inputs:
        - **index_pred** (Tensor) - The predicted value of the nino3.4 index.
        Tensor of shape :math:`(batch\_size, obs\_time + pred\_time)`.
        - **index_true** (Tensor) - The true value of the nino3.4 index.
        Tensor of shape :math:`(batch\_size, obs\_time + pred\_time)`.

    Outputs:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from src import WeightedLoss
        >>> index_pred = Tensor(np.zeros([16, 32]), mindspore.float32)
        >>> index_true = Tensor(np.ones([16, 32]), mindspore.float32)
        >>> loss_fn = WeightedLoss(1, 0.02, 0.6, 12)
        >>> loss = loss_fn(index_pred, index_true)
        >>> print(loss)
        1.3199999
    """

    def __init__(self,
                 lambda_1=1.,
                 lambda_2=0.02,
                 lambda_3=0.6,
                 corr_point=0.5,
                 obs_time=12
                 ):
        super(WeightedLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.corr_point = corr_point
        self.obs_time = obs_time
        self.regloss = nn.MSELoss()

    def construct(self, index_pred, index_true):
        """WeightedLoss forward function"""
        pred = index_pred[:, self.obs_time:]
        gtrue = index_true[:, self.obs_time:]
        regloss1 = self.regloss(pred, gtrue)

        if self.lambda_2 != 0:
            regloss2 = self.regloss(index_pred[:, :self.obs_time], index_true[:, :self.obs_time])
        else:
            regloss2 = 0

        if self.lambda_3 != 0:
            pred_ = Tensor.sub(pred.t(), Tensor.mean(pred, axis=1))
            gtrue_ = Tensor.sub(gtrue.t(), Tensor.mean(gtrue, axis=1))
            corr = self.corr_point - ops.cosine_similarity(pred_, gtrue_, dim=1)
            corr = ops.maximum(corr, ops.zeros_like(corr, dtype=ms.float32))
            corr_loss = Tensor.mean(corr)
        else:
            corr_loss = 0

        return regloss1 * self.lambda_1\
               + regloss2 * self.lambda_2 \
               + corr_loss * self.lambda_3


class InferenceModule:
    """
    Perform model inference.
    """
    def __init__(self, model, config, logger):
        self.model = model
        self.data_params = config.get("data")
        self.logger = logger

    def eval(self, valid_dataset):
        """Get correlation coefficient of validation dataset"""
        self.model.set_train(False)
        pred_list = []
        true_list = []
        for inputs, outputs in valid_dataset:
            pred = self.model(inputs)
            pred_list.append(pred)
            true_list.append(outputs)
        pred = ops.cat(pred_list, axis=0)
        true = ops.cat(true_list, axis=0)
        output = pred.T.asnumpy()
        target = true.T.asnumpy()
        corr_list = []
        for index_month in range(output.shape[0]):
            corr, _ = sps.pearsonr(output[index_month], target[index_month])
            corr_list.append(corr)
        obs_acc = 100. * np.mean(corr_list[:self.data_params.get('obs_time')])
        pred_acc = 100. * np.mean(corr_list[self.data_params.get('obs_time'):])
        print('OBS Accuracy: {:.2f}%, Pred Accuracy: {:.2f}%, Pred Corr:'.format(obs_acc, pred_acc))
        print(np.round(corr_list[self.data_params.get('obs_time'):], 2))
        return corr_list


class EvaluateCallBack(Callback):
    """
    Monitor the prediction accuracy in training.
    """
    def __init__(self,
                 model,
                 valid_dataset,
                 config,
                 logger,
                 ):
        super(EvaluateCallBack, self).__init__()
        self.config = config
        summary_params = config.get('summary')
        self.summary_dir = summary_params.get('summary_dir')
        self.predict_interval = summary_params.get('eval_interval')
        self.logger = logger
        self.valid_dataset = valid_dataset
        self.eval_net = InferenceModule(model, config, logger=self.logger)
        self.eval_time = 0

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def on_eval(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            self.eval_time += 1
            self.eval_net.eval(self.valid_dataset)
