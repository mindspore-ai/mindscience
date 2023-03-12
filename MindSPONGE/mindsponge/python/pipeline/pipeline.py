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
import ssl
import urllib.request
import mindspore as ms
from mindspore import context
from mindsponge.common.config_load import load_config
from .models import Multimer, MultimerDataSet, multimer_configuration
from .models import COLABDESIGN, ColabDesignDataSet, colabdesign_configuration
from .models import KGNN, KGNNDataSet, kgnn_configuration
from .models import UFold, UFoldDataSet, ufold_configuration
from .models import DeepDR, DeepDRDataSet, deepdr_configuration
from .models import MEGAFold, MEGAFoldDataSet, megafold_configuration

model_card = {
    "MEGAFold": {"model": MEGAFold, "dataset": MEGAFoldDataSet, "config": megafold_configuration},
    "Multimer": {"model": Multimer, "dataset": MultimerDataSet, "config": multimer_configuration},
    "ColabDesign": {"model": COLABDESIGN, "dataset": ColabDesignDataSet, "config": colabdesign_configuration},
    "KGNN": {"model": KGNN, "dataset": KGNNDataSet, "config": kgnn_configuration},
    "UFold": {"model": UFold, "dataset": UFoldDataSet, "config": ufold_configuration},
    "DeepDR": {"model": DeepDR, "dataset": DeepDRDataSet, "config": deepdr_configuration}
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

    def initialize(self, key=None, config_path=None, **kwargs):
        if config_path is None:
            config = download_config(self.config[key], self.config_path + key + ".yaml")
        else:
            config = load_config(config_path)
        self.model = self.model_cls(config, **kwargs)
        self.dataset = self.dataset_cls(config)

    def set_device_id(self, device_id):
        context.set_context(device_id=device_id)

    def predict(self, data, **kwargs):
        data = self.dataset.process(data, **kwargs)
        result = self.model.predict(data, **kwargs)
        return result

    def train(self, data_source, num_epochs=1, **kwargs):
        self.dataset.set_training_data_src(data_source, **kwargs)
        data_iter = self.dataset.create_iterator(num_epochs, **kwargs)
        for _ in range(num_epochs):
            for d in data_iter:
                loss = self.model.train_step(d)
                print(loss)

    def save_model(self, ckpt_path=None):
        if ckpt_path is not None:
            ms.save_checkpoint(self.model.network, ckpt_path)
        else:
            print("Checkpoint path is None!")
