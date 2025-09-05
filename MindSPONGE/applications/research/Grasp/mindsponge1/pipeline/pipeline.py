# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""Pipeline"""
import os
import time
import ssl
import urllib.request
from mindspore import context
from ..common.config_load import load_config
from .models import Multimer, MultimerDataSet, multimer_configuration
from .models import COLABDESIGN, ColabDesignDataSet, colabdesign_configuration

model_card = {
    "Multimer": {"model": Multimer, "dataset": MultimerDataSet, "config": multimer_configuration},
    "ColabDesign": {"model": COLABDESIGN, "dataset": ColabDesignDataSet, "config": colabdesign_configuration},
}


def download_config(url, save_path):
    if not os.path.exists(save_path):
        prefix, _ = os.path.split(save_path)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        print("Download config to ", save_path)
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, save_path)
    config = load_config(save_path)
    return config


class PipeLine:
    """PipeLine"""

    def __init__(self, name):
        self.model_cls = model_card[name]["model"]
        self.dataset_cls = model_card[name]["dataset"]
        self.config = model_card[name]["config"]
        self.model = None
        self.dataset = None
        self.config_path = "./config/"

    def initialize(self, key):
        config = download_config(self.config[key], self.config_path + key + ".yaml")
        self.model = self.model_cls(config)
        self.dataset = self.dataset_cls(config)

    def set_device_id(self, device_id):
        context.set_context(device_id=device_id)

    def predict(self, data):
        data = self.dataset.process(data)
        result = self.model.predict(data)
        return result

    def train(self, data_source, num_epochs):
        self.dataset.set_training_data_src(data_source)
        data_iter = self.dataset.create_iterator(num_epochs)
        for _ in range(num_epochs):
            for d in data_iter:
                loss = self.model.train_step(d)
                print(loss)

    def _test_predict(self, config, run_times=2):
        self.initialize(config)
        test_data = self.dataset._test_data_parse()
        for i in range(run_times):
            t1 = time.time()
            result = self.predict(test_data)
            t2 = time.time()
            print("predict times : ", i, " cost : ", t2 - t1, " s")
