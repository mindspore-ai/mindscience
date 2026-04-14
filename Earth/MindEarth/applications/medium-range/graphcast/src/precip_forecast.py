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
"""Precipitation Forecast"""
import os

import numpy as np

from mindearth.module.forecast import WeatherForecast


class WeatherForecastTp(WeatherForecast):
    """
    Self-defined WeatherForecast for precipitation inherited from `WeatherForecast`.
        Args:
        model (mindspore.nn.Cell): the network for training.
        config (dict): the configurations of model, dataset, train details, etc.
        logger (logging.RootLogger): Logger of the training process.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    @staticmethod
    def _get_history_climate_mean(config):
        """get history climate mean."""
        data_params = config.get('data')
        climate_mean = np.load(os.path.join(data_params.get("root_dir"), "statistic",
                                            f"climate_{data_params.get('grid_resolution')}_tp.npy"))

        return climate_mean

    def _print_key_metrics(self, rmse, acc, rqe=None):
        """print key info metrics"""
        for timestep in self.config['summary']['key_info_timestep']:
            self.logger.info(f't = {timestep} hour: ')
            timestep_idx = int(timestep) // self.pred_lead_time - 1
            tp_rmse = rmse[timestep_idx]
            tp_acc = acc[timestep_idx]
            tp_rqe = rqe[timestep_idx]
            self.logger.info(f" RMSE of TP: {tp_rmse}")
            self.logger.info(f" ACC  of TP: {tp_acc}")
            self.logger.info(f" RQE  of TP: {tp_rqe}")
