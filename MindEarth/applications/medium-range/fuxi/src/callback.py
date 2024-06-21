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
"""callback for fuxi"""
import os
import time
import numpy as np

import mindspore as ms
from mindspore import ops, nn
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord
from mindearth.module import WeatherForecast

from .utils import plt_key_info


class MAELossForMultiLabel(nn.LossBase):
    """ MAELossForMultiLabel definition """

    def __init__(self, data_params=None, optimizer_params=None):
        super(MAELossForMultiLabel, self).__init__()
        self.abs = ops.Abs()
        self.data_params = data_params
        self.batch_size = data_params.get('batch_size', 1)
        self.optimizer_params = optimizer_params
        self.loss_weight = optimizer_params.get("loss_weight", 0.25)
        self.h_size = data_params.get("h_size", 720)
        self.w_size = data_params.get("w_size", 1440)
        self.feature_dims = data_params.get("feature_dims", 69)
        self.level_feature_size = data_params.get("level_feature_size", 5)
        self.pressure_level_num = data_params.get("pressure_level_num", 13)
        self.surface_feature_size = self.feature_dims - self.level_feature_size * self.pressure_level_num

    def get_loss(self, x):
        """
        Computes the loss.
        Args:
            x (Tensor)
        Returns:
            Return the loss.
        """
        input_dtype = x.dtype
        x = ops.cast(x, ms.float32)
        x = ops.mean(x)
        x = ops.cast(x, input_dtype)
        return x
    def construct(self, x, x_surface, labels):
        """MAELossForMultiLabel forward function."""
        label = labels[..., :-self.surface_feature_size]
        label_surface = labels[..., -self.surface_feature_size:]
        label = label.reshape(self.batch_size, self.h_size, self.w_size, self.level_feature_size,
                              self.pressure_level_num)
        label = label.transpose(0, 3, 4, 1, 2)
        label_surface = label_surface.reshape(self.batch_size, self.h_size, self.w_size, -1)
        label_surface = label_surface.transpose(0, 3, 1, 2)
        x1 = self.abs(x - label)
        x2 = self.abs(x_surface - label_surface)
        loss_x = self.get_loss(x1)
        loss_surface = self.get_loss(x2)
        loss = loss_x + self.loss_weight * loss_surface
        return loss


class CustomWithLossCell(nn.Cell):
    """
    custom loss
    """

    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, labels):
        """Custom loss forward function"""
        pred, pred_surface = self._backbone(data)
        loss = self._loss_fn(pred, pred_surface, labels)
        return loss


class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.
    """

    def __init__(self, model, config, logger):
        super(InferenceModule, self).__init__(model, config, logger)
        self.model = model
        data_params = config.get('data')
        self.h_size = data_params.get('h_size', 720)
        self.w_size = data_params.get('w_size', 1440)
        self.level_feature_size = data_params.get('level_feature_size', 5)
        self.pressure_level_num = data_params.get('pressure_level_num', 13)
        self.surface_feature_size = data_params.get('surface_feature_size', 4)
        self.feature_dims = data_params.get('feature_dims', 69)
        self.logger = logger
        self.batch_size = data_params.get('batch_size', 1)
        self.t_in = data_params.get('t_in', 1)
        climate_mean = np.load(os.path.join(data_params.get("root_dir"), "statistic",
                                            f"climate_{data_params.get('grid_resolution')}.npy"))
        self.climate_mean = climate_mean.reshape(-1, self.w_size, self.feature_dims)[:-1].reshape(-1, self.feature_dims)

    def forecast(self, inputs):
        """InferenceModule forecast"""
        pred_list, data_list = [], []
        for _ in range(self.t_out):
            inputs = inputs.squeeze()
            cur_infer_start_time = time.time()
            pred, pred_surface = self.model(inputs)
            cur_infer_end_time = time.time()
            print(f"out shape:{pred.shape}, time cost:{cur_infer_end_time - cur_infer_start_time}")
            pred = pred.reshape(self.batch_size, self.level_feature_size * self.pressure_level_num,
                                self.h_size * self.w_size)
            pred_surface = pred_surface.reshape(self.batch_size, self.surface_feature_size, self.h_size * self.w_size)
            all_pred = ops.concat((pred, pred_surface), axis=1).transpose(0, 2, 1)
            pred_list.append(all_pred.asnumpy())
            if self.t_out > 1:
                if self.t_in == 1:
                    inputs = all_pred
                else:
                    data_list = [inputs[..., self.feature_dims:self.feature_dims * self.t_in], all_pred]
                    inputs = ops.concat(data_list, axis=-1).reshape(-1, self.feature_dims * self.t_in)
        return pred_list


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

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            self.eval_time += 1
            lat_weight_rmse, lat_weight_acc = self.eval_net.eval(self.valid_dataset, generator_flag=True)
            summary_params = self.config.get('summary')
            if summary_params.get('plt_key_info'):
                plt_key_info(lat_weight_rmse, self.config, self.eval_time * self.predict_interval, metrics_type="RMSE",
                             loc="upper left")
                plt_key_info(lat_weight_acc, self.config, self.eval_time * self.predict_interval, metrics_type="ACC",
                             loc="lower left")
