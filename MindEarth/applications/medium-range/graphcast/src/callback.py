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
import math

import numpy as np
import mindspore.numpy as mnp
from mindspore import dtype as mstype
from mindspore import ops, nn
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord

from mindearth.module import WeatherForecast
from .utils import plt_key_info, unlog_trans
from .precip_forecast import WeatherForecastTp


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
                data_list = [inputs[..., feature_dims:feature_dims * t_in], pred]
                inputs = ops.concat(data_list, axis=-1).reshape(-1, feature_dims * t_in)
    return pred_list


def _forecast_tp_multi_step(inputs, model, feature_dims, t_out, t_in):
    """Precipitation forecast multiple steps with given inputs"""
    pred_list, data_list = [], []
    for _ in range(t_out):
        pred, pred_recon = model(inputs)
        pred = ops.cast(pred, inputs.dtype)
        pred_recon = ops.cast(pred_recon, inputs.dtype)
        inputs = inputs.squeeze()
        pred_list.append(pred)
        if t_out > 1:
            if t_in == 1:
                inputs = pred_recon
            else:
                data_list = [inputs[..., feature_dims:feature_dims * t_in], pred_recon]
                inputs = ops.concat(data_list, axis=-1).reshape(-1, feature_dims * t_in)
    return pred_list


class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.
    """

    def __init__(self, model, config, logger):
        super(InferenceModule, self).__init__(model, config, logger)
        self.model = model
        data_params = config.get('data')
        self.w_size = data_params.get('w_size')
        self.h_size = data_params.get('h_size')
        self.feature_dims = data_params.get('feature_dims')
        self.t_out_test = data_params.get('t_out_test')
        self.logger = logger
        self.batch_size = data_params.get('batch_size')
        self.t_in = data_params.get('t_in')

    def forecast(self, inputs):
        pred_list = _forecast_multi_step(inputs, self.model, self.feature_dims, self.t_out_test, self.t_in)
        return pred_list

    def _get_lat_weight(self):
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(math.pi / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        return weight

    def _get_metrics(self, inputs, labels):
        """Get lat_weight_rmse and lat_weight_acc metrics"""
        pred = self.forecast(inputs)
        pred = ops.stack(pred, axis=0).asnumpy()
        labels = labels.asnumpy()

        lat_weight_rmse_step = self._calculate_lat_weighted_error(labels, pred).transpose()
        lat_weight_acc = self._calculate_lat_weighted_acc(labels, pred).transpose()

        return lat_weight_rmse_step, lat_weight_acc

    def _calculate_lat_weighted_error(self, label, prediction):
        """calculate latitude weighted error"""
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1, 1)
        error = np.square(label[0] - prediction) # the index 0 of label shape is batch_size
        lat_weight_error = np.sum(error * grid_node_weight, axis=1)
        return lat_weight_error

    def _calculate_lat_weighted_acc(self, label, prediction):
        """calculate latitude weighted acc"""
        prediction = prediction * self.total_std.reshape((1, 1, -1)) + self.total_mean.reshape((1, 1, -1))
        label = label * self.total_std.reshape((1, 1, 1, -1)) + self.total_mean.reshape((1, 1, 1, -1))
        prediction = prediction - self.climate_mean
        label = label - self.climate_mean
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(1, -1, 1)
        acc_numerator = np.sum(prediction * label[0] * grid_node_weight, axis=1)
        acc_denominator = np.sqrt(np.sum(prediction ** 2 * grid_node_weight,
                                         axis=1) * np.sum(label[0] ** 2 * grid_node_weight, axis=1))
        try:
            acc = acc_numerator / acc_denominator
        except ZeroDivisionError as e:
            print(repr(e))
        return acc


class InferenceModuleTp(WeatherForecastTp):
    """
    Perform multiple rounds of precipitation model inference.
    """

    def __init__(self, model, config, logger):
        super().__init__(model, config, logger)
        self.model = model
        data_params = config.get('data')
        self.w_size = data_params.get("w_size", 720)
        self.h_size = data_params.get("h_size", 360)
        self.feature_dims = data_params.get("feature_dims", 69)
        self.t_out_test = data_params.get("t_out_test", 20)
        self.logger = logger
        self.batch_size = data_params.get("batch_size", 1)
        self.t_in = data_params.get("t_in", 1)
        self.resolution = data_params.get("grid_resolution", 0.5)
        self.percentile = (1. - np.logspace(-1, -4, 50)) * 100
        self.tp_unit_trans = 1000

    def _get_metrics(self, inputs, labels):
        """Get lat_weight_rmse, lat_weight_acc and rqe metrics"""
        pred = self.forecast(inputs)
        labels = unlog_trans(labels).asnumpy()
        pred = unlog_trans(pred).asnumpy()
        lat_weight_rmse_step = self._calculate_lat_weighted_error(labels * self.tp_unit_trans,
                                                                  pred * self.tp_unit_trans)
        rqe_step = self._calculate_rqe(labels, pred)
        labels = labels - self.climate_mean
        pred = pred - self.climate_mean
        lat_weight_acc_step = self._calculate_lat_weighted_acc(labels, pred)
        return lat_weight_acc_step, lat_weight_rmse_step, rqe_step

    def _calculate_lat_weight(self):
        """Get latitude weight"""
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(math.pi / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        grid_lat_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1)
        return grid_lat_weight.astype(np.float32)

    def _calculate_rqe(self, label, prediction):
        """Get rqe"""
        label = label.reshape(self.t_out_test, -1)
        prediction = prediction.reshape(self.t_out_test, -1)
        quantile_label = np.percentile(label, self.percentile, axis=1)
        quantile_pred = np.percentile(prediction, self.percentile, axis=1)
        try:
            rqe = np.mean((quantile_pred - quantile_label) / quantile_label, axis=0)
        except ZeroDivisionError:
            return np.zeros(self.t_out_test)
        return rqe

    def _calculate_lat_weighted_error(self, label, prediction):
        """Get rmse"""
        batch_size = label.shape[0]
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(math.pi / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(1, 1, -1)
        error = np.square(np.reshape(label, (batch_size, self.t_out_test, -1)) - np.reshape(prediction, (
            batch_size, self.t_out_test, -1)))
        lat_weight_error = np.sum(error * grid_node_weight, axis=(0, 2))
        return lat_weight_error

    def _calculate_lat_weighted_acc(self, label, prediction):
        """Get acc"""
        batch_size = label.shape[0]
        lat_t = np.arange(0, self.h_size)

        s = np.sum(np.cos(math.pi / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(1, 1, -1)
        pred_prime = np.reshape(prediction, (batch_size, self.t_out_test, -1))
        label_prime = np.reshape(label, (batch_size, self.t_out_test, -1))
        a = np.sum(pred_prime * label_prime * grid_node_weight, axis=(0, 2))
        b = np.sqrt(
            np.sum(pred_prime ** 2 * grid_node_weight, axis=(0, 2)) * np.sum(label_prime ** 2 * grid_node_weight,
                                                                             axis=(0, 2)))
        try:
            acc = a / b
        except ZeroDivisionError:
            return np.zeros(self.t_out_test)
        return acc

    def forecast(self, inputs):
        """Get the precipitation forecast"""
        pred_list = _forecast_tp_multi_step(inputs, self.model, self.feature_dims, self.t_out_test, self.t_in)
        pred_list = ops.concat(pred_list, axis=1)
        return pred_list

    def eval(self, dataset):
        """
        Eval the model using test dataset or validation dataset.

        Args:
            dataset (mindspore.dataset): The dataset for eval, including inputs and labels.
        """
        self.logger.info("================================Start Evaluation================================")
        data_length = 0
        eval_data_length = 0
        lat_weight_rmse = 0.
        lat_weight_acc = 0.
        rqe = 0.
        for data in dataset.create_dict_iterator():
            inputs = data['inputs']
            batch_size = inputs.shape[0]
            labels = data['labels']
            lat_weight_acc_step, lat_weight_rmse_step, rqe_step = self._get_metrics(inputs, labels)
            lat_weight_acc += lat_weight_acc_step
            lat_weight_rmse += lat_weight_rmse_step
            rqe += rqe_step
            eval_data_length += batch_size
            self.logger.info(eval_data_length)
        self.logger.info(f'test dataset size: {data_length}')
        try:
            lat_weight_acc = lat_weight_acc / eval_data_length
            lat_weight_rmse = np.sqrt(
                lat_weight_rmse / (eval_data_length * self.w_size * self.h_size))
            rqe = rqe / eval_data_length
        except ZeroDivisionError:
            return np.zeros(self.t_out_test), np.zeros(self.t_out_test), np.zeros(self.t_out_test)
        self._print_key_metrics(lat_weight_rmse, lat_weight_acc, rqe)
        self.logger.info(lat_weight_acc)
        self.logger.info("================================End Evaluation================================")
        return lat_weight_rmse, lat_weight_acc, rqe


class LossNet(nn.Cell):
    """ LossNet definition """

    def __init__(self, ai, wj, sj_std, feature_dims, tp=False):
        super().__init__()
        self.feature_dims = feature_dims
        self.err_weight = wj * ai / sj_std
        self.tp = tp

    def construct(self, label, pred):
        """
        forward function.
        if tp is True, this function will calculate LP Loss.
        """
        if self.tp:
            batch_size = pred.shape[0]
            diff_norms = ops.norm(pred.reshape((batch_size, -1)) - label.reshape((batch_size, -1)), dim=1, ord=2.0)
            y_norms = ops.norm(label.reshape((batch_size, -1)), dim=1, ord=2.0)
            loss = ops.div(diff_norms, y_norms)
            return loss.mean()
        pred = ops.cast(pred, mstype.float32)
        label = ops.squeeze(label[..., :self.feature_dims])
        pred = ops.squeeze(pred)
        err = mnp.square(pred - label)
        weighted_err = err * self.err_weight
        weighted_err = mnp.reshape(weighted_err, (pred.shape[-2], -1))
        loss = mnp.average(weighted_err)
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
        summary_params = config.get('summary')
        data_params = config.get('data')
        self.data_params = data_params
        self.summary_dir = summary_params.get('summary_dir')
        self.predict_interval = summary_params.get('eval_interval')
        self.logger = logger
        self.valid_dataset = valid_dataset
        if data_params.get("tp", False):
            self.eval_net_tp = InferenceModuleTp(model, config, logger=self.logger)
        else:
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
            if self.data_params.get("tp", False):
                lat_weight_rmse, lat_weight_acc, _ = self.eval_net_tp.eval(self.valid_dataset)
            else:
                lat_weight_rmse, lat_weight_acc = self.eval_net.eval(self.valid_dataset)
                if self.config['summary']['plt_key_info']:
                    plt_key_info(lat_weight_rmse, self.config, self.eval_time * self.predict_interval,
                                 metrics_type="RMSE", loc="upper left")
                    plt_key_info(lat_weight_acc, self.config, self.eval_time * self.predict_interval,
                                 metrics_type="ACC", loc="lower left")


class CustomWithLossCell(nn.Cell):
    """
    custom loss
    """

    def __init__(self, backbone, loss_fn, data_params):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

        self.feature_dims = data_params.get('feature_dims')
        self.t_out_train = data_params.get('t_out_train')
        self.t_in = data_params.get('t_in')
        self.tp = data_params.get("tp", False)

    def construct(self, data, labels):
        """Custom loss forward function"""
        if self.tp:
            pred_list = _forecast_tp_multi_step(data, self._backbone, self.feature_dims, self.t_out_train, self.t_in)
        else:
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
