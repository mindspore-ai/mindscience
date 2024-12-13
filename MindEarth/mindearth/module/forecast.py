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
import time
from threading import Thread

import math
import numpy as np
from mindspore import Tensor

from ..data import FEATURE_DICT, SIZE_DICT

PI = math.pi


class WeatherForecast:
    """
    Base class of Weather Forecast model inference.
    All user-define forecast model should be inherited from this class during inference.
    This class can be called in the callback of the trainer or during inference through loading the checkpoint.
    By calling this class, the model can perform inference based on the input model using the custom forecast member
    function. t_out defines the number of forward inference passes to be made by the model.

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
        >>>         for t in range(self.t_out):
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
        ...         'save_checkpoint_epochs': 1,
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
        self.model = model
        self.amp_level = config['train']['amp_level']
        self.logger = logger
        self.config = config
        self.adjust_size = False
        self.data_params = config['data']
        self.weather_data_source = self.data_params.get('name', 'era5')
        self.grid_resolution = config['data'].get('grid_resolution', 1.4)
        self.h_size, self.w_size = SIZE_DICT[self.grid_resolution]
        if self.config['data'].get('patch', False):
            self.patch_size = [self.config['data'].get('patch_size', 8)]
            self.adjust_size = self.h_size % self.patch_size[0] != 0
            self.h_size = self.h_size - self.h_size % self.patch_size[0]
        self.feature_dims = config['data'].get('feature_dims', 69)
        self.total_std = self._get_total_sample_description(config, "std")
        self.total_mean = self._get_total_sample_description(config, "mean")
        if config['model']['name'] == "GraphCastTp":
            self.climate_mean = self._get_history_climate_mean(config)
        else:
            self.climate_mean = self._get_history_climate_mean(config, self.w_size, self.adjust_size)
        self.run_mode = config['train'].get("run_mode", 'train')
        if self.run_mode == 'train':
            self.t_out = config['data'].get("t_out_valid", 20)
        else:
            self.t_out = config['data'].get("t_out_test", 20)
        self.pred_lead_time = config['data']['pred_lead_time']
        self.batch_size = self.data_params['batch_size']

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
    def _get_history_climate_mean(config, w_size=None, adjust_size=False):
        """get history climate mean."""
        data_params = config.get('data')
        climate_mean = np.load(os.path.join(data_params.get("root_dir"), "statistic",
                                            f"climate_{data_params.get('grid_resolution')}.npy"))
        feature_dims = climate_mean.shape[-1]
        if adjust_size:
            climate_mean = climate_mean.reshape(-1, w_size, feature_dims)[:-1].reshape(-1, feature_dims)

        return climate_mean

    @staticmethod
    def forecast(inputs, labels=None):
        """
        The forecast function of the model.

        Args:
            inputs (Tensor): The input data of model.
            labels (Tensor): True values of the samples.
        """
        raise NotImplementedError("forecast module not implemented")

    def eval(self, dataset, generator_flag=False):
        '''
        Eval the model using test dataset or validation dataset.

        Args:
            dataset (mindspore.dataset): The dataset for eval, including inputs and labels.
            generator_flag (bool): "generator_flag" is used to pass a parameter to the "compute_total_rmse_acc" method.
                A flag indicating whether to use a data generator or not.
        '''
        data_length = len(dataset) // self.batch_size
        self.logger.info("================================Start Evaluation================================")
        self.logger.info(f'test dataset size: {data_length}')
        lat_weight_acc, lat_weight_rmse = self.compute_total_rmse_acc(dataset, generator_flag)
        self._print_key_metrics(lat_weight_rmse, lat_weight_acc)
        self._save_rmse_acc(lat_weight_rmse, lat_weight_acc)
        self.logger.info("================================End Evaluation================================")
        return lat_weight_rmse, lat_weight_acc

    def compute_total_rmse_acc(self, dataset, generator_flag):
        """
        Compute the total Root Mean Square Error (RMSE) and Accuracy for the dataset.

        This function iterates over the dataset, calculates the RMSE and accuracy for
        each batch, and accumulates the results to compute the total RMSE and accuracy
        over the entire dataset.

        Args:
            dataset (Dataset): The dataset object to compute metrics for.
            generator_flag (bool): A flag indicating whether to use a data generator or not.

        Returns:
            A tuple containing the total accuracy and RMSE for the dataset.

        Raises:
            NotImplementedError: If an unsupported data source is specified.
        """
        data_length = 0
        lat_weight_rmse = None
        lat_weight_acc = None
        if not generator_flag:
            data_iterator = dataset.create_dict_iterator()

        while data_length < len(dataset) // self.batch_size * self.batch_size:
            inputs_list, labels_list = [], []
            data_index = data_length
            if generator_flag:
                for _ in range(self.batch_size):
                    data = dataset.gen_data(data_index)
                    inputs_list.append(np.expand_dims(data[0], 0))
                    labels_list.append(np.expand_dims(data[1], 0))
                    data_index += 1
                inputs = np.concatenate(inputs_list)
                labels = np.concatenate(labels_list)
                inputs = self.get_inputs_data(inputs)
            else:
                data = next(data_iterator)
                inputs = data['inputs']
                labels = data['labels']
            lat_weight_rmse_step, lat_weight_acc_step = self._get_metrics(inputs, labels)
            if data_length == 0:
                lat_weight_rmse = lat_weight_rmse_step
                lat_weight_acc = lat_weight_acc_step
            else:
                lat_weight_rmse += lat_weight_rmse_step
                lat_weight_acc += lat_weight_acc_step
            data_length += self.batch_size
        return lat_weight_acc / data_length, lat_weight_rmse / data_length

    def get_inputs_data(self, inputs):
        return Tensor(inputs)

    def _get_metrics(self, inputs, labels):
        """get metrics for plot"""
        start_time = time.time()
        pred = self.forecast(inputs)
        print('PREDICTION TIME:', time.time() - start_time, 's')

        start_time = time.time()
        lat_weight_rmse = np.zeros((self.batch_size, self.feature_dims, self.t_out))
        lat_weight_acc = np.zeros((self.batch_size, self.feature_dims, self.t_out))
        threads = []
        for batch_index in range(self.batch_size):
            for t in range(self.t_out):
                thread = Thread(target=self._task,
                                args=(labels, pred, t, batch_index, lat_weight_rmse, lat_weight_acc))
                threads.append(thread)
                thread.start()
        for thread in threads:
            thread.join()
        print('COMPUTE RMSE AND ACC TIME:', time.time() - start_time, 's')

        lat_weight_rmse = np.sum(lat_weight_rmse, axis=0)
        lat_weight_acc = np.sum(lat_weight_acc, axis=0)
        return lat_weight_rmse, lat_weight_acc

    def get_key_metrics_index_list(self):
        z500_idx = self._get_absolute_idx(FEATURE_DICT.get("Z500"))
        t2m_idx = self._get_absolute_idx(FEATURE_DICT.get("T2M"))
        t850_idx = self._get_absolute_idx(FEATURE_DICT.get("T850"))
        u10_idx = self._get_absolute_idx(FEATURE_DICT.get("U10"))
        return [z500_idx, t2m_idx, t850_idx, u10_idx]

    def _save_rmse_acc(self, denormalized_lat_weight_rmse, lat_weight_acc):
        if self.config.get("summary").get("save_rmse_acc"):
            np.save(os.path.join(self.config.get("summary").get("summary_dir"),
                                 "denormalized_lat_weight_rmse.npy"), denormalized_lat_weight_rmse)
            np.save(os.path.join(self.config.get("summary").get("summary_dir"),
                                 "lat_weight_acc.npy"), lat_weight_acc)

    def _get_absolute_idx(self, idx):
        return idx[1] * self.config['data']['pressure_level_num'] + idx[0]

    def _print_key_metrics(self, rmse, acc):
        """print key info metrics"""
        z500_idx, t2m_idx, t850_idx, u10_idx = self.get_key_metrics_index_list()
        for timestep in self.config['summary']['key_info_timestep']:
            self.logger.info(f't = {timestep} hour: ')
            timestep_idx = int(timestep) // self.pred_lead_time - 1
            self.logger.info(f" RMSE of Z500: {rmse[z500_idx, timestep_idx]}, "
                             f"T2m: {rmse[t2m_idx, timestep_idx]}, "
                             f"T850: {rmse[t850_idx, timestep_idx]}, "
                             f"U10: {rmse[u10_idx, timestep_idx]}")
            self.logger.info(f" ACC  of Z500: {acc[z500_idx, timestep_idx]}, "
                             f"T2m: {acc[t2m_idx, timestep_idx]}, "
                             f"T850: {acc[t850_idx, timestep_idx]}, "
                             f"U10: {acc[u10_idx, timestep_idx]}")

    def _lat(self, j):
        return 90. - j * 180. / float(self.h_size - 1)

    def _latitude_weighting_factor(self, j, s):
        return self.h_size * np.cos(math.pi / 180. * self._lat(j)) / s

    def _get_lat_weight(self):
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(PI / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        return weight

    def _task(self, labels, pred, t, batch_index, lat_weight_rmse, lat_weight_acc):
        prediction = pred[t][batch_index:batch_index + 1, ...].squeeze()
        label = labels[batch_index:batch_index + 1, t, :, :].squeeze()
        lat_weight_rmse[batch_index, :, t] = self._calculate_lat_weighted_rmse(
            label, prediction)
        lat_weight_acc[batch_index, :, t] = self._calculate_lat_weighted_acc(
            label, prediction)

    def _calculate_lat_weighted_rmse(self, label, prediction):
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(1, -1)
        error = np.square(label - prediction).transpose(1, 0)
        lat_weight_error = np.sum(error * grid_node_weight, axis=1)
        lat_weight_rmse = np.sqrt(
            lat_weight_error / (self.w_size * self.h_size))
        lat_weight_rmse = lat_weight_rmse * self.total_std.squeeze()
        return lat_weight_rmse

    def _calculate_lat_weighted_acc(self, label, prediction):
        """calculate latitude weighted acc"""
        total_std = self.total_std.squeeze()
        total_mean = self.total_mean.squeeze()
        prediction = prediction * total_std + total_mean
        label = label * total_std + total_mean
        prediction = prediction - self.climate_mean
        label = label - self.climate_mean
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(1, -1, 1)
        acc_numerator = np.sum(prediction * label * grid_node_weight, axis=1)
        acc_denominator = np.sqrt(np.sum(prediction ** 2 * grid_node_weight,
                                         axis=1) * np.sum(label ** 2 * grid_node_weight,
                                                          axis=1))
        try:
            acc = acc_numerator / acc_denominator
        except ZeroDivisionError as e:
            print(repr(e))
        return acc
