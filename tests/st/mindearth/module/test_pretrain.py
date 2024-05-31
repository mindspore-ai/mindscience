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
"""test mindearth Trainer"""
import os

import numpy as np
import pytest

from mindspore import context, nn
import mindspore.dataset as ds

from mindearth.core import RelativeRMSELoss
from mindearth.module import Trainer
from mindearth.utils import load_yaml_config, create_logger


class Net(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(in_channels, out_channels, has_bias=False)

    def construct(self, x):
        x = self.fc1(x)
        return x


#pylint: disable=R1705
class MyIterable:
    """Self-defined dataset generator."""
    def __init__(self):
        self._index = 0
        self._data = np.random.rand(1, 32768, 69).astype(np.float32)
        self._label = np.random.rand(1, 32768, 69).astype(np.float32)

    def __next__(self):
        if self._index < len(self._data):
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item
        else:
            raise StopIteration

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self._data)


class MyTrainer(Trainer):
    """Self-defined Trainer"""
    def __init__(self, config, model, loss_fn, logger, feature_dims):
        super(MyTrainer, self).__init__(config, model, loss_fn, logger)
        self.feature_dims = feature_dims

    def get_callback(self):
        return 0

    def get_dataset(self):
        dataset = ds.GeneratorDataset(source=MyIterable(), column_names=["inputs", "labels"])
        return dataset, dataset


@pytest.mark.level0
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_trainer():
    """
    Feature: Test Trainer in platform gpu and ascend.
    Description: The training task should run normally.
    Expectation: Success or throw AssertionError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    file_path = os.path.abspath(__file__)
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(file_path), "..", "test_config.yaml"))
    config = load_yaml_config(yaml_path)
    logger = create_logger("./log.log")
    model = Net(69, 69)
    loss_fn = RelativeRMSELoss()
    trainer = MyTrainer(config, model, loss_fn, logger, 69)
    trainer.train()
