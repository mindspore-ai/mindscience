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
import os
import math
import numpy as np

from mindearth import WeatherForecast, FEATURE_DICT
from mindspore import Tensor
from mindspore.train.callback import Callback

from .utils import unpatchify, plt_key_info

PI = math.pi


class InferenceModule(WeatherForecast):
    """
    Perform multiple rounds of model inference.

    Args:
    """

    def __init__(self, model, config, logger):
        super(InferenceModule, self).__init__(model, config, logger)
        self.model = model
        self.config = config
        self.feature_dims = config['data'].get('feature_dims', 69)
        self.use_pretrain_model = (self.feature_dims == 20 and self.grid_resolution == 0.25)
        if self.use_pretrain_model:
            self.test_data_dir = config['data']['test_data_dir']
            self.climate_mean_path = os.path.join(self.test_data_dir, 'climate_mean_20feats.npy')
            self.climate_mean = np.load(self.climate_mean_path).astype(np.float32).reshape(self.h_size * self.w_size,
                                                                                           -1)
            self.total_std = np.load(os.path.join(self.test_data_dir, 'global_stds.npy'))[:, :-1, :, :].astype(
                np.float32)
            self.total_mean = np.load(os.path.join(self.test_data_dir, 'global_means.npy'))[:, :-1, :, :].astype(
                np.float32)

    def get_inputs_data(self, inputs):
        if self.use_pretrain_model:
            inputs = Tensor(inputs.squeeze(0))
        elif self.feature_dims == 69:
            inputs = Tensor(inputs)
        return inputs

    def forecast(self, inputs):
        pred_lst = []
        for _ in range(self.t_out):
            pred = self.model(inputs)
            pred = unpatchify(pred, (self.h_size, self.w_size),
                              self.config.get('data').get('patch_size'))
            pred_lst.append(pred.asnumpy().reshape(self.batch_size, self.h_size * self.w_size, self.feature_dims))
            inputs = pred.transpose(0, 3, 1, 2)
        return pred_lst

    def get_key_metrics_index_list(self):
        if self.use_pretrain_model:
            return [-6, 2, 5, 0]
        z500_idx = self._get_absolute_idx(FEATURE_DICT.get("Z500"))
        t2m_idx = self._get_absolute_idx(FEATURE_DICT.get("T2M"))
        t850_idx = self._get_absolute_idx(FEATURE_DICT.get("T850"))
        u10_idx = self._get_absolute_idx(FEATURE_DICT.get("U10"))
        return [z500_idx, t2m_idx, t850_idx, u10_idx]


class EvaluateCallBack(Callback):
    """
    Monitor the prediction accuracy in training.

    Args:
    """

    def __init__(self,
                 model,
                 valid_dataset_generator,
                 config,
                 logger
                 ):
        super(EvaluateCallBack, self).__init__()
        self.config = config
        self.eval_time = 0
        self.model = model
        self.valid_dataset_generator = valid_dataset_generator
        self.predict_interval = config.get('summary').get("valid_frequency")
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
            self.eval_time += 1
            lat_weight_rmse, lat_weight_acc = self.eval_net.eval(self.valid_dataset_generator, generator_flag=True)
            if self.config.get('summary').get('plt_key_info'):
                plt_key_info(lat_weight_rmse, self.config, self.eval_time * self.predict_interval, metrics_type="RMSE",
                             loc="upper left")
                plt_key_info(lat_weight_acc, self.config, self.eval_time * self.predict_interval, metrics_type="ACC",
                             loc="lower left")
