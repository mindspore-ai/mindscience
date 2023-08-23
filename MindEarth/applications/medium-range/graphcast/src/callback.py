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
"""earth with loss"""
import numpy as np

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import dtype as mstype
from mindspore import ops, nn
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord

from mindearth.module import WeatherForecast

from .utils import plt_key_info


def _forecast_multi_step(inputs, model, feature_dims, t_out, t_in):
    """Forecast multiple steps with given inputs"""
    pred_list, data_list = [], []
    for _ in range(t_out):
        pred = ops.cast(model(inputs), inputs.dtype)
        inputs = inputs.squeeze()
        pred_list.append(pred)
        if t_out > 1:
            if t_in == 1:
                inputs = pred
            else:
                data_list = (inputs[..., feature_dims:feature_dims * t_in], pred)
                inputs = ops.concat(data_list, axis=-1).reshape(-1, feature_dims * t_in)
    return pred_list


class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.
    """

    def __init__(self, model, config, logger):
        super(InferenceModule, self).__init__(model, config, logger)
        self.model = model
        self.w_size = config['data']['w_size']
        self.h_size = config['data']['h_size']
        self.feature_dims = config['data']['feature_dims']
        self.t_out_test = config['data']['t_out_test']
        self.logger = logger
        self.batch_size = config['data']['batch_size']
        self.t_in = config['data']['t_in']

    def _get_metrics(self, inputs, labels):
        """Get lat_weight_rmse and lat_weight_acc metrics"""
        pred = self.forecast(inputs)
        pred = ops.stack(pred, 0).reshape(self.batch_size, self.t_out_test, self.h_size * self.w_size,
                                          self.feature_dims)
        pred = ops.cast(pred, ms.float32)

        # rmse
        error = ops.square(pred - labels).transpose(0, 1, 3, 2).reshape(
            self.batch_size * self.t_out_test * self.feature_dims, -1)
        weight = ms.Tensor(self._calculate_lat_weight().reshape(-1, 1))
        lat_weight_rmse_step = ops.matmul(error, weight)
        lat_weight_rmse_step = lat_weight_rmse_step.reshape(self.t_out_test, self.feature_dims).transpose(1,
                                                                                                          0).asnumpy()

        # acc
        acc_numerator = pred * labels
        acc_numerator = acc_numerator.transpose(0, 1, 3, 2).reshape(
            self.batch_size * self.t_out_test * self.feature_dims, -1)
        acc_numerator = ops.matmul(acc_numerator, weight)

        pred_square = ops.square(pred).transpose(0, 1, 3, 2).reshape(
            self.batch_size * self.t_out_test * self.feature_dims, -1)
        label_square = ops.square(labels).transpose(0, 1, 3, 2).reshape(
            self.batch_size * self.t_out_test * self.feature_dims, -1)

        acc_denominator = ops.sqrt(ops.matmul(pred_square, weight) * ops.matmul(label_square, weight))
        lat_weight_acc = acc_numerator / acc_denominator
        lat_weight_acc_step = lat_weight_acc.reshape(self.t_out_test, self.feature_dims).transpose(1, 0).asnumpy()

        return lat_weight_rmse_step, lat_weight_acc_step

    def _calculate_lat_weight(self):
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(3.1416 / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        grid_lat_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1)
        return grid_lat_weight.astype(np.float32)

    def forecast(self, inputs):
        pred_list = _forecast_multi_step(inputs, self.model, self.feature_dims, self.t_out_test, self.t_in)
        return pred_list


class LossNet(nn.Cell):
    """ LossNet definition """

    def __init__(self, ai, wj, sj_std, feature_dims):
        super().__init__()
        self.sj_std = sj_std
        self.wj = wj
        self.ai = ai
        self.feature_dims = feature_dims
        self.err_weight = self.wj * self.ai / self.sj_std

    def construct(self, label, pred):
        pred = ops.cast(pred, mstype.float32)
        label = ops.squeeze(label[..., :self.feature_dims])
        pred = ops.squeeze(pred)
        err = msnp.square(pred - label)
        weighted_err = err * self.err_weight
        weighted_err = msnp.reshape(weighted_err, (pred.shape[-2], -1))
        loss = msnp.average(weighted_err)
        return loss


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
        summary_params = config['summary']
        self.summary_dir = summary_params['summary_dir']
        self.predict_interval = summary_params['eval_interval']
        self.logger = logger
        self.valid_dataset = valid_dataset
        self.eval_net = InferenceModule(model, config, logger=self.logger)
        self.eval_time = 0

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            self.eval_time += 1
            lat_weight_rmse, lat_weight_acc = self.eval_net.eval(self.valid_dataset)
            if self.config['summary']['plt_key_info']:
                plt_key_info(lat_weight_rmse, self.config, self.eval_time * self.predict_interval, metrics_type="RMSE",
                             loc="upper left")
                plt_key_info(lat_weight_acc, self.config, self.eval_time * self.predict_interval, metrics_type="ACC",
                             loc="lower left")


class CustomWithLossCell(nn.Cell):
    """
    custom loss
    """

    def __init__(self, backbone, loss_fn, data_params):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

        self.feature_dims = data_params['feature_dims']
        self.t_out_train = data_params['t_out_train']
        self.t_in = data_params['t_in']

    def construct(self, data, labels):
        """Custom loss forward function"""
        pred_list = _forecast_multi_step(data, self._backbone, self.feature_dims, self.t_out_train, self.t_in)
        loss = 0
        for t in range(self.t_out_train):
            pred = pred_list[t]
            if self.t_out_train == 1:
                label = ops.squeeze(labels)
            else:
                label = ops.squeeze(labels[:, t])
            loss_step = self._loss_fn(label, pred)
            loss += loss_step
        loss = loss / self.t_out_train
        return loss
