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
"""test mindearth WeatherForecast"""
import os
import numpy as np
import pytest

from mindspore import nn, context, ops

from mindearth.utils import load_yaml_config, create_logger
from mindearth.data import Dataset
from mindearth.module import WeatherForecast


class Net(nn.Cell):
    """Self-defined net for test."""
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(in_channels, out_channels, has_bias=False)

    def construct(self, x):
        x = self.fc1(x)
        return x


#pylint: disable=R1720
class MyIterable:
    """Self-defined dataset generator."""
    def __init__(self):
        self._index = 0
        self._data = np.random.rand(1, 32768, 69).astype(np.float32)
        self._label = np.random.rand(1, 32768, 69).astype(np.float32)

    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)


#pylint: disable=W0613
class MyInference(WeatherForecast):
    """Self-defined WeatherForecast"""
    def __init__(self, model, config, logger):
        super(MyInference, self).__init__(model, config, logger)
        self.total_mean = np.random.rand(69,).astype(np.float32)

    def forecast(self, inputs):
        pred_lst = []
        for _ in range(self.t_out_test):
            pred = self.model(inputs)
            pred_lst.append(pred)
            inputs = pred
        return pred_lst

    @staticmethod
    def _get_total_sample_description(config, info_mode="std"):
        return np.random.rand(69,).astype(np.float32)

    @staticmethod
    def _get_history_climate_mean(config, w_size, adjust_size=False):
        return np.random.rand(32768, 69).astype(np.float32)

    def _get_metrics(self, inputs, labels):
        pred = self.forecast(inputs)
        pred = ops.stack(pred, axis=0).asnumpy()
        pred = pred[0]
        labels = labels.asnumpy()
        labels = np.expand_dims(labels, axis=0)
        lat_weight_rmse_step = self._calculate_lat_weighted_error(labels, pred).transpose()
        lat_weight_acc = self._calculate_lat_weighted_acc(labels, pred).transpose()

        return lat_weight_rmse_step, lat_weight_acc

    def _get_lat_weight(self):
        lat_t = np.arange(0, self.h_size)
        s = np.sum(np.cos(3.1416 / 180. * self._lat(lat_t)))
        weight = self._latitude_weighting_factor(lat_t, s)
        return weight

    def _calculate_lat_weighted_error(self, label, prediction):
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(-1, 1)
        error = np.square(label[0] - prediction)
        lat_weight_error = np.sum(error * grid_node_weight, axis=1)
        return lat_weight_error

    def _calculate_lat_weighted_acc(self, label, prediction):
        prediction = prediction * self.total_std.reshape((1, 1, -1)) + self.total_mean.reshape((1, 1, -1))
        label = label * self.total_std.reshape((1, 1, 1, -1)) + self.total_mean.reshape((1, 1, 1, -1))
        weight = self._get_lat_weight()
        grid_node_weight = np.repeat(weight, self.w_size, axis=0).reshape(1, -1, 1)
        a = np.sum(prediction * label[0] * grid_node_weight, axis=1)
        b = np.sqrt(np.sum(prediction ** 2 * grid_node_weight, axis=1) * np.sum(label[0] ** 2 * grid_node_weight,
                                                                                axis=1))
        acc = a / b
        return acc


@pytest.mark.level0
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_forecast():
    """
    Feature: Test WeatherForecast in platform gpu and ascend.
    Description: The forecast task should run normally.
    Expectation: Success or throw AssertionError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    file_path = os.path.abspath(__file__)
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(file_path), "..", "test_config.yaml"))
    config = load_yaml_config(yaml_path)
    logger = create_logger("./log.log")
    model = Net(69, 69)
    infer_module = MyInference(model, config, logger)
    dataset = Dataset(MyIterable())
    test_dataset = dataset.create_dataset(1)
    infer_module.eval(test_dataset)
