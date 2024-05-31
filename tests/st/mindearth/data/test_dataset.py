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
"""test mindearth Era5Data DemData RadarData"""

import os

import h5py
import numpy as np
import pytest

from mindearth.data import DemData, RadarData, Era5Data, Dataset
from mindearth.utils import load_yaml_config, make_dir


class MyEra5Data(Era5Data):
    """Self-defined Era5Data"""
    def _get_statistic(self):
        self.mean_pressure_level = np.random.rand(13, 1, 1, 5).astype(np.float32)
        self.std_pressure_level = np.random.rand(13, 1, 1, 5).astype(np.float32)
        self.mean_surface = np.random.rand(4,).astype(np.float32)
        self.std_surface = np.random.rand(4,).astype(np.float32)


@pytest.mark.level0
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_era5data():
    """
    Feature: Test Era5Data in platform gpu and ascend.
    Description: The Era5Data output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    file_path = os.path.abspath(__file__)
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(file_path), "..", "test_config.yaml"))
    data_path = os.path.abspath(os.path.join(os.path.dirname(file_path), "..", "test_data"))
    train_dir = os.path.join(data_path, 'train', '2015')
    train_surface_dir = os.path.join(data_path, 'train_surface', '2015')
    train_static_dir = os.path.join(data_path, 'train_static', '2015')
    train_surface_static_dir = os.path.join(data_path, 'train_surface_static', '2015')
    for data_dir in [train_dir, train_static_dir, train_surface_dir, train_surface_static_dir]:
        if not os.path.exists(data_dir):
            make_dir(data_dir)
    surface_data = np.random.rand(1, 128, 256, 4)
    level_data = np.random.rand(1, 13, 128, 256, 5).astype(np.float32)
    for file_name in ['2015_01_01_1.npy', '2015_01_01_7.npy', '2015_01_01_13.npy']:
        np.save(os.path.join(train_surface_dir, file_name), surface_data)
        np.save(os.path.join(train_dir, file_name), level_data)
    train_static = np.ones((5, 2)).astype(np.float32)
    train_static[:, 1] = 0.0
    np.save(os.path.join(train_static_dir, '2015.npy'), train_static)
    train_surface_static = np.ones((4, 2)).astype(np.float32)
    train_surface_static[:, 1] = 0.0
    np.save(os.path.join(train_surface_static_dir, '2015.npy'), train_surface_static)
    config = load_yaml_config(yaml_path)
    config['data']['root_dir'] = data_path
    data_params = config['data']
    dataset_gen = MyEra5Data(data_params=data_params, run_mode='train')
    dataset = Dataset(dataset_gen)
    test_dataset = dataset.create_dataset(1)
    len_data = test_dataset.get_dataset_size()
    assert len_data == 1, f"The dataset only provides 1 piece of data, but got {len_data}."
    for data in test_dataset.create_dict_iterator():
        x = data['inputs']
        assert x.shape == (1, 32768, 69), f"The input shape should be (1, 32768, 69), but got {x.shape}."
        label = data['labels']
        assert label.shape == (1, 32768, 69), f"The label shape should be (1, 32768, 69), but got {label.shape}."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_demdata():
    """
    Feature: Test DemData in platform gpu and ascend.
    Description: The DemData output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    file_path = os.path.abspath(__file__)
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(file_path), "..", "test_dem_config.yaml"))
    data_path = os.path.abspath(os.path.join(os.path.dirname(file_path), "dataset_dem"))
    train_dir = os.path.join(data_path, "train")
    make_dir(train_dir)
    train_32_32 = np.random.rand(10, 1, 32, 32)
    train_160_160 = np.random.rand(10, 1, 160, 160)
    f = h5py.File(os.path.join(train_dir, "train.h5"), "w")
    f.create_dataset(name='32_32', data=train_32_32)
    f.create_dataset(name='160_160', data=train_160_160)
    f.close()
    config = load_yaml_config(yaml_path)
    config['data']['root_dir'] = data_path
    data_params = config['data']
    dataset_gen = DemData(data_params)
    dataset = Dataset(dataset_gen)
    test_dataset = dataset.create_dataset(1)
    for data in test_dataset.create_dict_iterator():
        x = data["inputs"]
        assert x.shape == (1, 1, 32, 32), f"The input shape should be (1, 1, 32, 32), but got {x.shape}."
        label = data["labels"]
        assert label.shape == (1, 1, 160, 160), f"The label shape should be (1, 1, 160, 160), but got {label.shape}."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_radardata():
    """
    Feature: Test RadarData in platform gpu and ascend.
    Description: The RadarData output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    file_path = os.path.abspath(__file__)
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(file_path), "..", "test_dem_config.yaml"))
    data_path = os.path.abspath(os.path.join(os.path.dirname(file_path), "dataset_radar"))
    train_dir = os.path.join(data_path, "train")
    make_dir(train_dir)
    data = np.random.rand(24, 256, 256, 1)
    np.save(os.path.join(train_dir, "train.npy"), data)
    config = load_yaml_config(yaml_path)
    config['data']['root_dir'] = data_path
    data_params = config['data']
    dataset_gen = RadarData(data_params)
    dataset = Dataset(dataset_gen)
    test_dataset = dataset.create_dataset(1)
    for data in test_dataset.create_dict_iterator():
        x = data["inputs"]
        assert x.shape == (1, 4, 1, 256, 256), f"The input shape should be (1, 4, 1, 256, 256), but got {x.shape}."
        label = data["labels"]
        assert label.shape == (1, 18, 1, 256, 256), \
            f"The input shape should be (1, 18, 1, 256, 256), but got {x.shape}."
