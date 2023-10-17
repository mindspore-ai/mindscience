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
"""The callback of fourcastnet"""
import numpy as np
from mindspore.train.callback import Callback
from mindearth.module import WeatherForecast

from .utils import unpatchify, plt_key_info


class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.

    Args:
    """

    def __init__(self, model, config, logger):
        super(InferenceModule, self).__init__(model, config, logger)
        self.model = model

    def forecast(self, inputs):
        pred_lst = []
        for _ in range(self.t_out_test):
            pred = self.model(inputs)
            pred = unpatchify(pred, (self.h_size, self.w_size),
                              self.config.get('data').get('patch_size'))
            pred = pred.transpose(0, 3, 1, 2)
            pred_lst.append(pred)
            inputs = pred
        return pred_lst

    def _get_metrics(self, inputs, labels):
        """get metrics for plot"""
        pred = self.forecast(inputs)
        feature_num = labels.shape[1]
        lat_weight_rmse = np.zeros((feature_num, self.t_out_test))
        lat_weight_acc = np.zeros((feature_num, self.t_out_test))
        for t in range(self.t_out_test):
            for f in range(feature_num):
                lat_weight_rmse[f, t] = self._calculate_lat_weighted_rmse(
                    labels[:, f, t].asnumpy(), pred[t][:, f].asnumpy())  # label(B,C,T,H W) pred(B,C,H W)
                lat_weight_acc[f, t] = self._calculate_lat_weighted_acc(
                    labels[:, f, t].asnumpy(), pred[t][:, f].asnumpy())
        return lat_weight_rmse, lat_weight_acc


class EvaluateCallBack(Callback):
    """
    Monitor the prediction accuracy in training.

    Args:
    """

    def __init__(self,
                 model,
                 valid_dataset,
                 config,
                 logger
                 ):
        super(EvaluateCallBack, self).__init__()
        self.config = config
        self.eval_time = 0
        self.model = model
        self.valid_dataset = valid_dataset
        self.predict_interval = config.get('summary').get("valid_frequency")
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
            self.eval_time += 1
            lat_weight_rmse, lat_weight_acc = self.eval_net.eval(
                self.valid_dataset)
            if self.config.get('summary').get('plt_key_info'):
                plt_key_info(lat_weight_rmse, self.config, self.eval_time * self.predict_interval, metrics_type="RMSE",
                             loc="upper left")
                plt_key_info(lat_weight_acc, self.config, self.eval_time * self.predict_interval, metrics_type="ACC",
                             loc="lower left")
