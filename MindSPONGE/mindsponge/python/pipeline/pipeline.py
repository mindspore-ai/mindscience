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
from .models import COLABDESIGN, ColabDesignDataSet, colabdesign_configuration
from .models import DeepDR, DeepDRDataSet, deepdr_configuration
from .models import DeepFri, DeepFriDataSet, deepfri_configuration
from .models import ESM, ESMDataSet, esm_configuration
from .models import ESM2, ESM2DataSet, esm2_configuration
from .models import Grover, GroverDataSet, grover_configuration
from .models import KGNN, KGNNDataSet, kgnn_configuration
from .models import MEGAAssessment, MEGAAssessmentDataSet, megaassessment_configuration
from .models import MEGAEvoGen, MEGAEvoGenDataSet, megaevogen_configuration
from .models import MEGAFold, MEGAFoldDataSet, megafold_configuration
from .models import Multimer, MultimerDataSet, multimer_configuration
from .models import ProteinMpnn, ProteinMpnnDataset, proteinmpnn_configuration
from .models import UFold, UFoldDataSet, ufold_configuration

model_card = {
    "ColabDesign": {"model": COLABDESIGN, "dataset": ColabDesignDataSet, "config": colabdesign_configuration},
    "DeepDR": {"model": DeepDR, "dataset": DeepDRDataSet, "config": deepdr_configuration},
    "DeepFri": {"model": DeepFri, "dataset": DeepFriDataSet, "config": deepfri_configuration},
    "ESM_IF1": {"model": ESM, "dataset": ESMDataSet, "config": esm_configuration},
    "ESM2": {"model": ESM2, "dataset": ESM2DataSet, "config": esm2_configuration},
    "Grover": {"model": Grover, "dataset": GroverDataSet, "config": grover_configuration},
    "KGNN": {"model": KGNN, "dataset": KGNNDataSet, "config": kgnn_configuration},
    "MEGAAssessment": {"model": MEGAAssessment, "dataset": MEGAAssessmentDataSet,
                       "config": megaassessment_configuration},
    "MEGAEvoGen": {"model": MEGAEvoGen, "dataset": MEGAEvoGenDataSet, "config": megaevogen_configuration},
    "MEGAFold": {"model": MEGAFold, "dataset": MEGAFoldDataSet, "config": megafold_configuration},
    "Multimer": {"model": Multimer, "dataset": MultimerDataSet, "config": multimer_configuration},
    "Proteinmpnn": {"model": ProteinMpnn, "dataset": ProteinMpnnDataset, "config": proteinmpnn_configuration},
    "UFold": {"model": UFold, "dataset": UFoldDataSet, "config": ufold_configuration},
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

    def initialize(self, key=None, conf=None, config_path=None, **kwargs):
        """initialize"""

        if conf is None:
            if config_path is not None:
                if config_path.endswith(".yaml"):
                    config = load_config(config_path)
                else:
                    self.config_path = config_path
                    config = download_config(self.config[key], self.config_path + key + ".yaml")
            else:
                config = download_config(self.config[key], self.config_path + key + ".yaml")
        else:
            config = conf
        self.model = self.model_cls(config, **kwargs)
        self.dataset = self.dataset_cls(config)

    def set_device_id(self, device_id):
        """set device id"""

        context.set_context(device_id=device_id)

    def predict(self, data, **kwargs):
        """predict"""

        data = self.dataset.process(data, **kwargs)
        result = self.model.predict(data, **kwargs)
        return result

    def train(self, data_source, num_epochs=1, **kwargs):
        """train"""

        self.dataset.set_training_data_src(data_source, **kwargs)
        data_iter = self.dataset.create_iterator(num_epochs, **kwargs)
        for _ in range(num_epochs):
            for d in data_iter:
                loss = self.model.train_step(d)
                print(loss)

    def save_model(self, ckpt_path=None):
        """save model"""

        if ckpt_path is not None:
            ms.save_checkpoint(self.model.network, ckpt_path)
        else:
            print("Checkpoint path is None!")
