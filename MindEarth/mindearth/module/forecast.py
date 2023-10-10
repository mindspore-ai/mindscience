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
"""Weather Forecast"""
import os
import math

import numpy as np

from mindspore import amp

from ..data import FEATURE_DICT, SIZE_DICT

PI = math.pi


class WeatherForecast:
    """
    Base class of Weather Forecast model inference.
    All user-define forecast model should be inherited from this class during inference.
    This class can be called in the callback of the trainer or during inference through loading the checkpoint.
    By calling this class, the model can perform inference based on the input model using the custom forecast member
    function. t_out_test defines the number of forward inference passes to be made by the model.

    Args:
        model (mindspore.nn.Cell): the network for training.
        config (dict): the configurations of model, dataset, train details, etc.
        logger (logging.RootLogger): Logger of the training process.

    Note:
        - The member function, `forecast`, must be overridden to define the forward inference process of the model.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import logging
        >>> from mindspore import Tensor, nn
        >>> import mindspore
        >>> from mindearth.data import Era5Data,Dataset
        >>> from mindearth.module import WeatherForecast
        ...
        >>> class Net(nn.Cell):
        >>>     def __init__(self, in_channels, out_channels):
        >>>         super(Net, self).__init__()
        >>>         self.fc1 = nn.Dense(in_channels, 128, weight_init='ones')
        >>>         self.fc2 = nn.Dense(128, out_channels, weight_init='ones')
        ...
        >>>     def construct(self, x):
        >>>         x = x.transpose(0, 2, 3, 1)
        >>>         x = self.fc1(x)
        >>>         x = self.fc2(x)
        >>>         x = x.transpose(0, 3, 1, 2)
        >>>         return x
        ...
        >>> class InferenceModule(WeatherForecast):
        >>>     def __init__(self, model, config, logger):
        >>>         super(InferenceModule, self).__init__(model, config, logger)
        ...
        >>>     def forecast(self, inputs, labels=None):
        >>>         pred_lst = []
        >>>         for t in range(self.t_out_test):
        >>>             pred = self.model(inputs)
        >>>             pred_lst.append(pred)
        >>>             inputs = pred
        >>>         return pred_lst
        ...
        >>> config={
        ...     "model": {
        ...         'name': 'Net'
        ...     },
        ...     "data": {
        ...         'name': 'era5',
        ...         'root_dir': './dataset',
        ...         'feature_dims': 69,
        ...         't_in': 1,
        ...         't_out_train': 1,
        ...         't_out_valid': 20,
        ...         't_out_test': 20,
        ...         'valid_interval': 1,
        ...         'test_interval': 1,
        ...         'train_interval': 1,
        ...         'pred_lead_time': 6,
        ...         'data_frequency': 6,
        ...         'train_period': [2015, 2015],
        ...         'valid_period': [2016, 2016],
        ...         'test_period': [2017, 2017],
        ...         'patch': True,
        ...         'patch_size': 8,
        ...         'batch_size': 8,
        ...         'num_workers': 1,
        ...         'grid_resolution': 1.4,
        ...         'h_size': 128,
        ...         'w_size': 256
        ...     },
        ...     "optimizer": {
        ...         'name': 'adam',
        ...         'weight_decay': 0.0,
        ...         'epochs': 200,
        ...         'finetune_epochs': 1,
        ...         'warmup_epochs': 1,
        ...         'initial_lr': 0.0005
        ...     },
        ...     "summary": {
        ...         'save_checkpoint_steps': 1,
        ...         'keep_checkpoint_max': 10,
        ...         'valid_frequency': 10,
        ...         'summary_dir': '/path/to/summary',
        ...         'ckpt_path': '/path/to/ckpt'
        ...     },
        ...     "train": {
        ...         'name': 'oop',
        ...         'distribute': False,
        ...         'device_id': 2,
        ...         'amp_level': 'O2',
        ...         'run_mode': 'test',
        ...         'load_ckpt': True
        ...     }
        ... }
        ...
        >>> model = Net(in_channels = config['data']['feature_dims'], out_channels = config['data']['feature_dims'])
        >>> infer_module = InferenceModule(model, config,logging.getLogger())
        >>> test_dataset_generator = Era5Data(data_params=config['data'], run_mode='test')
        >>> test_dataset = Dataset(test_dataset_generator, distribute=config['train']['distribute'],
        ...                        num_workers = config['data']['num_workers'], shuffle=False)
        >>> test_dataset = test_dataset.create_dataset(1)
        >>> infer_module.eval(test_dataset)
    """

    def __init__(self,
                 model,
                 config,
                 logger
                 ):
        self.model = amp.auto_mixed_precision(model, config['train']['amp_level'])
        self.logger = logger
        self.config = config
        self.total_std = self._get_total_sample_description(config, "std")
        self.total_mean = self._get_total_sample_description(config, "mean")
        self.climate_mean = self._get_history_climate_mean(config)
        self.h_size, self.w_size = SIZE_DICT[config['data'].get('grid_resolution', 1.4)]
        self.t_out_test = config['data'].get("t_out_test", 20)
        self.pred_lead_time = config['data']['pred_lead_time']

    @staticmethod
    def _get_total_sample_description(config, info_mode):
        """get total sample std or mean description."""
        root_dir = config.get('data').get('root_dir')
        sample_info_pressure_levels = np.load(
            os.path.join(root_dir, "statistic", info_mode + ".npy"))
        sample_info_pressure_levels = sample_info_pressure_levels.transpose(1, 2, 3, 0)
        sample_info_pressure_levels = sample_info_pressure_levels.reshape((1, -1))
        sample_info_pressure_levels = np.squeeze(sample_info_pressure_levels, axis=0)
        sample_info_surface = np.load(os.path.join(root_dir, "statistic",
                                                   info_mode + "_s.npy"))
        total_sample_info = np.append(sample_info_pressure_levels, sample_info_surface)

        return total_sample_info

    @staticmethod
    def _get_history_climate_mean(config):
        """get history climate mean."""
        data_params = config.get('data')
        climate_mean = np.load(os.path.join(data_params.get("root_dir"), "statistic",
                                            f"climate_{data_params.get('grid_resolution')}.npy"))

        return climate_mean

    def _get_absolute_idx(self, idx):
        return idx[1] * self.config['data']['pressure_level_num'] + idx[0]

    def _print_key_metrics(self, rmse, acc):
        """print key info metrics"""
        z500_idx = self._get_absolute_idx(FEATURE_DICT.get("Z500"))
        t2m_idx = self._get_absolute_idx(FEATURE_DICT.get("T2M"))
        t850_idx = self._get_absolute_idx(FEATURE_DICT.get("T850"))
        u10_idx = self._get_absolute_idx(FEATURE_DICT.get("U10"))
        for timestep in self.config['summary']['key_info_timestep']:
            self.logger.info(f't = {timestep} hour: ')
            timestep_idx = int(timestep) // self.pred_lead_time - 1
            z500_rmse = rmse[z500_idx, timestep_idx]
            z500_acc = acc[z500_idx, timestep_idx]
            t2m_rmse = rmse[t2m_idx, timestep_idx]
            t2m_acc = acc[t2m_idx, timestep_idx]
            t850_rmse = rmse[t850_idx, timestep_idx]
            t850_acc = acc[t850_idx, timestep_idx]
            u10_rmse = rmse[u10_idx, timestep_idx]
            u10_acc = acc[u10_idx, timestep_idx]

            self.logger.info(f" RMSE of Z500: {z500_rmse}, T2m: {t2m_rmse}, T850: {t850_rmse}, U10: {u10_rmse}")
            self.logger.info(f" ACC  of Z500: {z500_acc}, T2m: {t2m_acc}, T850: {t850_acc}, U10: {u10_acc}")

    @staticmethod
    def forecast(inputs, labels=None):
        """
        The forecast function of the model.

        Args:
            inputs (Tensor): The input data of model.
            labels (Tensor): True values of the samples.
        """
        raise NotImplementedError("forecast module not implemented")

    def eval(self, dataset):
        '''
        Eval the model using test dataset or validation dataset.

        Args:
            dataset (mindspore.dataset): The dataset for eval, including inputs and labels.
        '''
        self.logger.info("================================Start Evaluation================================")
        data_length = 0
        lat_weight_rmse = []
        lat_weight_acc = []
        for data in dataset.create_dict_iterator():
            inputs = data['inputs']
            batch_size = inputs.shape[0]
            labels = data['labels']
            lat_weight_rmse_step, lat_weight_acc_step = self._get_metrics(inputs, labels)
            if data_length == 0:
                lat_weight_rmse = lat_weight_rmse_step
                lat_weight_acc = lat_weight_acc_step
            else:
                lat_weight_rmse += lat_weight_rmse_step
                lat_weight_acc += lat_weight_acc_step

            data_length += batch_size

        self.logger.info(f'test dataset size: {data_length}')
        # (69, 20)
        lat_weight_rmse = np.sqrt(
            lat_weight_rmse / (data_length * self.w_size * self.h_size))
        lat_weight_acc = lat_weight_acc / data_length
        temp_rmse = lat_weight_rmse.transpose(1, 0)
        denormalized_lat_weight_rmse = temp_rmse * self.total_std
        denormalized_lat_weight_rmse = denormalized_lat_weight_rmse.transpose(1, 0)
        if self.config.get("summary").get("save_rmse_acc"):
            np.save(os.path.join(self.config.get("summary").get("summary_dir"),
                                 "denormalized_lat_weight_rmse.npy"), denormalized_lat_weight_rmse)
            np.save(os.path.join(self.config.get("summary").get("summary_dir"),
                                 "lat_weight_acc.npy"), lat_weight_acc)
        self._print_key_metrics(denormalized_lat_weight_rmse, lat_weight_acc)

        self.logger.info("================================End Evaluation================================")
        return denormalized_lat_weight_rmse, lat_weight_acc

    def _get_metrics(self, inputs, labels):
        """get metrics for plot"""
        pred = self.forecast(inputs, labels)
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

    def _lat(self, j):
        return 90. - j * 180. / float(self.h_size - 1)

    def _latitude_weighting_factor(self, j, s):
        return self.h_size * np.cos(PI / 180. * self._lat(j)) / s

    def _calculate_lat_weighted_rmse(self, label, prediction):
        batch_size = label.shape[0]
        lat_t = np.arange(0, self.h_size)

        s = np.sum(np.cos(PI / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1)
        error = np.square(np.reshape(label, (batch_size, -1)) - np.reshape(prediction, (batch_size, -1)))
        lat_weight_error = np.sum(error * grid_node_weight)
        return lat_weight_error

    def _calculate_lat_weighted_acc(self, label, prediction):
        """ calculate lat weighted acc"""
        lat_t = np.arange(0, self.h_size)

        s = np.sum(np.cos(PI / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s).reshape(self.h_size, 1)
        grid_node_weight = np.repeat(weight, self.w_size, axis=1)

        pred_prime = prediction
        label_prime = label
        a = np.sum(pred_prime * label * grid_node_weight)
        b = np.sqrt(np.sum(pred_prime ** 2 * grid_node_weight) * np.sum(label_prime ** 2 * grid_node_weight))
        acc = a / b
        return acc
