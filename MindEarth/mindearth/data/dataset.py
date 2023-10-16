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
# ============================================================================
'''Module providing dataset functions'''
import os
import abc
import datetime
import random

import h5py
import numpy as np

import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size

from ..utils import get_datapath_from_date

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator as validator
except ImportError:
    import mindspore._checkparam as validator

# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002203
PRESSURE_LEVELS_WEATHERBENCH_13 = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# The list of all possible atmospheric variables. Taken from:
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Table9
ALL_ATMOSPHERIC_VARS = (
    "potential_vorticity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "vorticity",
    "divergence",
    "relative_humidity",
    "ozone_mass_mixing_ratio",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "fraction_of_cloud_cover",
)

TARGET_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
)
TARGET_SURFACE_NO_PRECIP_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
)
TARGET_ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
TARGET_ATMOSPHERIC_NO_W_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)

FEATURE_DICT = {'Z500': (7, 0), 'T850': (10, 2), 'U10': (-3, 0), 'T2M': (-1, 0)}
SIZE_DICT = {0.25: [721, 1440], 0.5: [360, 720], 1.4: [128, 256]}


class Data:
    """
    This class is the base class of Dataset.

    Args:
        root_dir (str, optional): The root dir of input data. Default: ".".

    Raises:
        TypeError: If the type of train_dir is not str.
        TypeError: If the type of test_dir is not str.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, root_dir="."):
        self.train_dir = os.path.join(root_dir, "train")
        self.valid_dir = os.path.join(root_dir, 'valid')
        self.test_dir = os.path.join(root_dir, "test")

    @abc.abstractmethod
    def __getitem__(self, index):
        """Defines behavior for when an item is accessed. Return the corresponding element for given index."""
        raise NotImplementedError(
            "{}.__getitem__ not implemented".format(self.dataset_type))

    @abc.abstractmethod
    def __len__(self):
        """Return length of dataset"""
        raise NotImplementedError(
            "{}.__len__ not implemented".format(self.dataset_type))


class Era5Data(Data):
    """
    This class is used to process ERA5 re-analyze data, and is used to generate the dataset generator supported by
    MindSpore. This class inherits the Data class.

    Args:
        data_params (dict): dataset-related configuration of the model.
        run_mode (str, optional): whether the dataset is used for training, evaluation or testing. Supports [“train”,
            “test”, “valid”]. Default: 'train'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindearth.data import Era5Data
        >>> data_params = {
        ...     'name': 'era5',
        ...     'root_dir': './dataset',
        ...     'feature_dims': 69,
        ...     't_in': 1,
        ...     't_out_train': 1,
        ...     't_out_valid': 20,
        ...     't_out_test': 20,
        ...     'valid_interval': 1,
        ...     'test_interval': 1,
        ...     'train_interval': 1,
        ...     'pred_lead_time': 6,
        ...     'data_frequency': 6,
        ...     'train_period': [2015, 2015],
        ...     'valid_period': [2016, 2016],
        ...     'test_period': [2017, 2017],
        ...     'patch': True,
        ...     'patch_size': 8,
        ...     'batch_size': 8,
        ...     'num_workers': 1,
        ...     'grid_resolution': 1.4,
        ...     'h_size': 128,
        ...     'w_size': 256
        ... }
        >>> dataset_generator = Era5Data(data_params)
    """

    ## TODO: example should include all possible infos:
    #  data_frequency, patch/patch_size
    def __init__(self,
                 data_params,
                 run_mode='train'):

        super(Era5Data, self).__init__(data_params.get('root_dir'))
        none_type = type(None)
        root_dir = data_params.get('root_dir')
        self.train_surface_dir = os.path.join(root_dir, "train_surface")
        self.valid_surface_dir = os.path.join(root_dir, "valid_surface")
        self.test_surface_dir = os.path.join(root_dir, "test_surface")

        self.train_static = os.path.join(root_dir, "train_static")
        self.valid_static = os.path.join(root_dir, "valid_static")
        self.test_static = os.path.join(root_dir, "test_static")
        self.train_surface_static = os.path.join(root_dir, "train_surface_static")
        self.valid_surface_static = os.path.join(root_dir, "valid_surface_static")
        self.test_surface_static = os.path.join(root_dir, "test_surface_static")

        self.statistic_dir = os.path.join(root_dir, "statistic")
        validator.check_value_type("train_dir", self.train_dir, [str, none_type])
        validator.check_value_type("test_dir", self.test_dir, [str, none_type])
        validator.check_value_type("valid_dir", self.valid_dir, [str, none_type])

        self._get_statistic()

        self.run_mode = run_mode
        self.t_in = data_params.get('t_in')
        self.h_size = data_params.get('h_size')
        self.w_size = data_params.get('w_size')
        self.data_frequency = data_params.get('data_frequency')
        self.valid_interval = data_params.get('valid_interval') * self.data_frequency
        self.test_interval = data_params.get('test_interval') * self.data_frequency
        self.train_interval = data_params.get('train_interval') * self.data_frequency
        self.pred_lead_time = data_params.get('pred_lead_time')
        self.train_period = data_params.get('train_period')
        self.valid_period = data_params.get('valid_period')
        self.test_period = data_params.get('test_period')
        self.feature_dims = data_params.get('feature_dims')
        self.output_dims = data_params.get('feature_dims')
        self.patch = data_params.get('patch')
        if self.patch:
            self.patch_size = data_params.get('patch_size')

        if run_mode == 'train':
            self.t_out = data_params.get('t_out_train')
            self.path = self.train_dir
            self.surface_path = self.train_surface_dir
            self.static_path = self.train_static
            self.static_surface_path = self.train_surface_static
            self.interval = self.train_interval
            self.start_date = datetime.datetime(self.train_period[0], 1, 1, 0, 0, 0)

        elif run_mode == 'valid':
            self.t_out = data_params['t_out_valid']
            self.path = self.valid_dir
            self.surface_path = self.valid_surface_dir
            self.static_path = self.valid_static
            self.static_surface_path = self.valid_surface_static
            self.interval = self.valid_interval
            self.start_date = datetime.datetime(self.valid_period[0], 1, 1, 0, 0, 0)

        else:
            self.t_out = data_params['t_out_test']
            self.path = self.test_dir
            self.surface_path = self.test_surface_dir
            self.static_path = self.test_static
            self.static_surface_path = self.test_surface_static
            self.interval = self.test_interval
            self.start_date = datetime.datetime(self.test_period[0], 1, 1, 0, 0, 0)

    def __len__(self):
        if self.run_mode == 'train':
            self.train_len = self._get_file_count(self.train_dir, self.train_period)
            length = (self.train_len * self.data_frequency -
                      (self.t_out + self.t_in) * self.pred_lead_time) // self.train_interval

        elif self.run_mode == 'valid':
            self.valid_len = self._get_file_count(self.valid_dir, self.valid_period)
            length = (self.valid_len * self.data_frequency -
                      (self.t_out + self.t_in) * self.pred_lead_time) // self.valid_interval

        else:
            self.test_len = self._get_file_count(self.test_dir, self.test_period)
            length = (self.test_len * self.data_frequency -
                      (self.t_out + self.t_in) * self.pred_lead_time) // self.test_interval
        return length

    def __getitem__(self, idx):
        inputs_lst = []
        inputs_surface_lst = []
        label_lst = []
        label_surface_lst = []
        idx = idx * self.interval

        for t in range(self.t_in):
            cur_input_data_idx = idx + t * self.pred_lead_time
            input_date, year_name = get_datapath_from_date(self.start_date, cur_input_data_idx.item())
            x = np.load(os.path.join(self.path, input_date)).astype(np.float32)
            x_surface = np.load(os.path.join(self.surface_path, input_date)).astype(np.float32)
            x_static = np.load(os.path.join(self.static_path, year_name)).astype(np.float32)
            x_surface_static = np.load(os.path.join(self.static_surface_path, year_name)).astype(np.float32)
            x = self._get_origin_data(x, x_static)
            x_surface = self._get_origin_data(x_surface, x_surface_static)
            x, x_surface = self._normalize(x, x_surface)
            inputs_lst.append(x)
            inputs_surface_lst.append(x_surface)

        for t in range(self.t_out):
            cur_label_data_idx = idx + (self.t_in + t) * self.pred_lead_time
            label_date, year_name = get_datapath_from_date(self.start_date, cur_label_data_idx.item())
            label = np.load(os.path.join(self.path, label_date)).astype(np.float32)
            label_surface = np.load(os.path.join(self.surface_path, label_date)).astype(np.float32)
            label_static = np.load(os.path.join(self.static_path, year_name)).astype(np.float32)
            label_surface_static = np.load(os.path.join(self.static_surface_path, year_name)).astype(np.float32)
            label = self._get_origin_data(label, label_static)
            label_surface = self._get_origin_data(label_surface, label_surface_static)
            label, label_surface = self._normalize(label, label_surface)

            label_lst.append(label)
            label_surface_lst.append(label_surface)

        x = np.squeeze(np.stack(inputs_lst, axis=0), axis=1).astype(np.float32)
        x_surface = np.squeeze(np.stack(inputs_surface_lst, axis=0), axis=1).astype(np.float32)
        label = np.squeeze(np.stack(label_lst, axis=0), axis=1).astype(np.float32)
        label_surface = np.squeeze(np.stack(label_surface_lst, axis=0), axis=1).astype(np.float32)
        return self._process_fn(x, x_surface, label, label_surface)

    @staticmethod
    def _get_origin_data(x, static):
        data = x * static[..., 0] + static[..., 1]
        return data

    @staticmethod
    def _get_file_count(path, period):
        file_lst = os.listdir(path)
        count = 0
        for f in file_lst:
            if period[0] <= int(f) <= period[1]:
                tmp_lst = os.listdir(os.path.join(path, f))
                count += len(tmp_lst)
        return count

    def _get_statistic(self):
        self.mean_pressure_level = np.load(os.path.join(self.statistic_dir, 'mean.npy'))
        self.std_pressure_level = np.load(os.path.join(self.statistic_dir, 'std.npy'))
        self.mean_surface = np.load(os.path.join(self.statistic_dir, 'mean_s.npy'))
        self.std_surface = np.load(os.path.join(self.statistic_dir, 'std_s.npy'))

    def _normalize(self, x, x_surface):
        x = (x - self.mean_pressure_level) / self.std_pressure_level
        x_surface = (x_surface - self.mean_surface) / self.std_surface
        return x, x_surface

    def _process_fn(self, x, x_surface, label, label_surface):
        '''process_fn'''
        _, level_size, _, _, feature_size = x.shape
        surface_size = x_surface.shape[-1]

        if self.patch:
            self.h_size = self.h_size - self.h_size % self.patch_size
            x = x[:, :, :self.h_size, ...]
            x_surface = x_surface[:, :self.h_size, ...]
            label = label[:, :, :self.h_size, ...]
            label_surface = label_surface[:, :self.h_size, ...]

            x = x.transpose((0, 4, 1, 2, 3)).reshape(self.t_in, level_size * feature_size, self.h_size, self.w_size)
            x_surface = x_surface.transpose((0, 3, 1, 2)).reshape(self.t_in, surface_size, self.h_size, self.w_size)
            label = label.transpose((0, 4, 1, 2, 3)).reshape(self.t_out, level_size * feature_size,
                                                             self.h_size, self.w_size)
            label_surface = label_surface.transpose((0, 3, 1, 2)).reshape(self.t_out, surface_size,
                                                                          self.h_size, self.w_size)
            inputs = np.concatenate([x, x_surface], axis=1)
            labels = np.concatenate([label, label_surface], axis=1)

        else:
            x = x.transpose((0, 2, 3, 4, 1)).reshape(self.t_in, self.h_size * self.w_size,
                                                     level_size * feature_size)
            x_surface = x_surface.reshape(self.t_in, self.h_size * self.w_size, surface_size)
            label = label.transpose((0, 2, 3, 4, 1)).reshape(self.t_out, self.h_size * self.w_size,
                                                             level_size * feature_size)
            label_surface = label_surface.reshape(self.t_out, self.h_size * self.w_size, surface_size)
            inputs = np.concatenate([x, x_surface], axis=-1)
            labels = np.concatenate([label, label_surface], axis=-1)
            inputs = inputs.transpose((1, 0, 2)).reshape(self.h_size * self.w_size,
                                                         self.t_in * (level_size * feature_size + surface_size))

        if self.patch:
            labels = self._patch(labels, (self.h_size, self.w_size), self.patch_size,
                                 level_size * feature_size + surface_size)
            inputs = np.squeeze(inputs)
        labels = np.squeeze(labels)
        return inputs, labels

    def _patch(self, x, img_size, patch_size, output_dims):
        """ Partition the data into patches. """
        if self.run_mode == 'train':
            x = x.transpose(0, 2, 3, 1)
            h, w = img_size[0] // patch_size, img_size[1] // patch_size
            x = x.reshape(x.shape[0], h, patch_size, w, patch_size, output_dims)
            x = x.transpose(0, 1, 3, 2, 4, 5)
            x = np.squeeze(x.reshape(x.shape[0], h * w, patch_size * patch_size * output_dims))
        else:
            x = x.transpose(1, 0, 2, 3)
        return x


class RadarData(Data):
    """
    This class is used to process dgmr radar data, and is used to generate the dataset generator supported by
    MindSpore. This class inherits the Data class.

    Args:
        data_params (dict): dataset-related configuration of the model.
        run_mode (str, optional): whether the dataset is used for training, evaluation or testing. Supports [“train”,
            “test”, “valid”]. Default: 'train'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindearth.data import RadarData
        >>> data_params = {
        ...     'name': 'radar',
        ...     'root_dir': './dataset',
        ...     'batch_size': 4,
        ...     'num_workers': 1,
        ...     't_out_train': '',
        ... }
        >>> dataset_generator = RadarData(data_params)
    """
    NUM_INPUT_FRAMES = 4
    NUM_TARGET_FRAMES = 18

    def __init__(self,
                 data_params,
                 run_mode='train'):
        super(RadarData, self).__init__(data_params.get("root_dir"))
        self.run_mode = run_mode
        if run_mode == 'train':
            file_list = os.walk(self.train_dir)
        elif run_mode == 'valid':
            file_list = os.walk(self.valid_dir)
        else:
            file_list = os.walk(self.test_dir)
        self.data = []
        for root, _, files in file_list:
            for file in files:
                if not file.endswith(".npy"):
                    continue
                json_path = os.path.join(root, file)
                self.data.append(json_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_dir = self.data[idx]
        with open(npy_dir, "rb") as file:
            radar_frames = np.load(file)
        if radar_frames is None:
            random.seed()
            new_idx = random.randint(0, len(self.data) - 1)
            return self.__getitem__(new_idx)

        input_frames = radar_frames[-RadarData.NUM_TARGET_FRAMES - RadarData.NUM_INPUT_FRAMES:
                                    -RadarData.NUM_TARGET_FRAMES]
        target_frames = radar_frames[-RadarData.NUM_TARGET_FRAMES:]
        return np.moveaxis(input_frames, [0, 1, 2, 3], [0, 2, 3, 1]), np.moveaxis(
            target_frames, [0, 1, 2, 3], [0, 2, 3, 1])


class DemData(Data):
    """
    This class is used to process Dem Super resolution data, and is used to generate the dataset generator supported by
    MindSpore. This class inherits the Data class.

    Args:
        data_params (dict): dataset-related configuration of the model.
        run_mode (str, optional): whether the dataset is used for training, evaluation or testing. Supports [“train”,
            “test”, “valid”]. Default: 'train'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindearth.data import DemData
        >>> data_params = {
        ...     'name': 'nasadem',
        ...     'root_dir': './dataset',
        ...     'patch_size': 32,
        ...     'batch_size': 64,
        ...     'epoch_size': 10,
        ...     'num_workers': 1,
        ...     't_out_train': '',
        ... }
        >>> dataset_generator = DemData(data_params)
    """

    def __init__(self,
                 data_params,
                 run_mode='train'):
        super(DemData, self).__init__(data_params['root_dir'])
        self.run_mode = run_mode

        if run_mode == 'train':
            path = os.path.join(self.train_dir, "train.h5")
        elif run_mode == 'valid':
            path = os.path.join(self.valid_dir, "valid.h5")
        else:
            path = os.path.join(self.test_dir, "test.h5")
        data = h5py.File(path, 'r')
        data_lr = data.get('32_32').astype(np.float32)
        data_hr = data.get('160_160').astype(np.float32)

        self.__data_lr = data_lr
        self.__data_hr = data_hr

    def __getitem__(self, index):
        return (self.__data_lr[index, :, :, :], self.__data_hr[index, :, :, :])

    def __len__(self):
        return len(self.__data_lr)


class Dataset:
    """
    Create the dataset for training, validation and testing,
    and output an instance of class mindspore.dataset.GeneratorDataset.

    Args:
        dataset_generator (Data): the data generator of weather dataset.
        distribute (bool, optional): whether or not to perform parallel training. Default: False.
        num_workers (int, optional): number of workers(threads) to process the dataset in parallel. Default: 1.
        shuffle (bool, optional): whether or not to perform shuffle on the dataset. Random accessible input is
                required. Default: True, expected order behavior shown in the table.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindearth.data import Era5Data, Dataset
        >>> data_params = {
        ...     'name': 'era5',
        ...     'root_dir': './dataset',
        ...     'feature_dims': 69,
        ...     't_in': 1,
        ...     't_out_train': 1,
        ...     't_out_valid': 20,
        ...     't_out_test': 20,
        ...     'valid_interval': 1,
        ...     'test_interval': 1,
        ...     'train_interval': 1,
        ...     'pred_lead_time': 6,
        ...     'data_frequency': 6,
        ...     'train_period': [2015, 2015],
        ...     'valid_period': [2016, 2016],
        ...     'test_period': [2017, 2017],
        ...     'patch': True,
        ...     'patch_size': 8,
        ...     'batch_size': 8,
        ...     'num_workers': 1,
        ...     'grid_resolution': 1.4,
        ...     'h_size': 128,
        ...     'w_size': 256
        ... }
        >>> dataset_generator = Era5Data(data_params)
        >>> dataset = Dataset(dataset_generator)
        >>> train_dataset = dataset.create_dataset(1)
    """

    def __init__(self,
                 dataset_generator, distribute=False, num_workers=1, shuffle=True):
        self.distribute = distribute
        self.num_workers = num_workers
        self.dataset_generator = dataset_generator
        self.shuffle = shuffle

        if distribute:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()

    def create_dataset(self, batch_size):
        """
        create dataset.

        Args:
            batch_size (int, optional): An int number of rows each batch is created with.

        Returns:
            BatchDataset, dataset batched.
        """
        ds.config.set_prefetch_size(1)
        dataset = ds.GeneratorDataset(self.dataset_generator,
                                      ['inputs', 'labels'],
                                      shuffle=self.shuffle,
                                      num_parallel_workers=self.num_workers)
        if self.distribute:
            distributed_sampler_train = ds.DistributedSampler(self.rank_size, self.rank_id)
            dataset.use_sampler(distributed_sampler_train)

        dataset_batch = dataset.batch(batch_size=batch_size, drop_remainder=True,
                                      num_parallel_workers=self.num_workers)
        return dataset_batch
