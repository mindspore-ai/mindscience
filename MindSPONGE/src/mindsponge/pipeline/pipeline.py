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
import logging
import urllib
import urllib.request
import mindspore as ms
from mindspore import context
from ..common.config_load import load_config
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
from .models import RASP, RASPDataSet, rasp_configuration
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
    "RASP": {"model": RASP, "dataset": RASPDataSet, "config": rasp_configuration},
    "UFold": {"model": UFold, "dataset": UFoldDataSet, "config": ufold_configuration},
}


extend_model = ["Pafnucy"]

def run_check(name):
    """check the valid of model name"""
    if name not in model_card and name not in extend_model:
        raise TypeError(f"The {name} model is not yet supported in this version of MindSPONGE,"
                        f" please select from {list(model_card.keys()) + extend_model}")
    if name in extend_model:
        try:
            # pylint: disable=W0611
            from openbabel import openbabel
            from openbabel import pybel
        # pylint: disable=W0703
        except Exception as e:
            # pylint: disable=W1203
            logging.error(f"{e}.\nOpen Babel is not correctly installed or version of Open Babel is lower than 3.0.0, "
                          "please refer to: https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/applications/"
                          "model_cards/pafnucy.md#%E4%BD%BF%E7%94%A8%E9%99%90%E5%88%B6")
        from .models.pafnucy import PAFNUCY, PAFNUCYDataSet, pafnucy_configuration
        extend_card = {"Pafnucy": {"model": PAFNUCY, "dataset": PAFNUCYDataSet, "config": pafnucy_configuration}}
        model_card.update(extend_card)

def download_config(url, save_path):
    """Download the config file"""
    if not os.path.exists(save_path):
        prefix, _ = os.path.split(save_path)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        logging.info("Download config to %s", save_path)
        try:
            # pylint: disable=W0212
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(url, save_path)
        except Exception as e:
            logging.error("Downloading from %s failed with %s.", url, str(e))
            raise e

class PipeLine:
    """PipeLine"""

    def __init__(self, name):
        run_check(name)
        self.model_cls = model_card[name]["model"]
        self.dataset_cls = model_card[name]["dataset"]
        self.config = model_card[name]["config"]
        self.model = None
        self.dataset = None
        self.config_path = "./config/"
        os.environ['MS_ASCEND_CHECK_OVERFLOW_MODE'] = "SATURATION_MODE"

    def set_config_path(self, config_path):
        self.config_path = config_path

    def initialize(self, key=None, conf=None, config_path=None, **kwargs):
        """initialize"""

        if sum(x is not None for x in (key, conf, config_path)) != 1:
            raise ValueError("Only one of key, conf, config_path can be not None")

        if conf is not None:
            logging.info("Initialize with user passed conf object")
            config = conf

        if config_path is not None:
            logging.info("Initialize with local config yaml file %s", config_path)
            config = load_config(config_path)

        if key is not None:
            logging.info("Initialize with standard config key %s", key)
            config_file = self.config_path + key + '.yaml'
            url = self.config.get(key)
            if not url:
                keys_supported = ', '.join(list(self.config.keys()))
                raise KeyError(f"User passed key {key} is not valid, valid keys are {keys_supported}")
            if os.path.exists(config_file):
                logging.warning("Using local config file for %s: %s", key, config_file)
            else:
                logging.warning("Local config file for %s not exist, download from %s", key, url)
                try:
                    download_config(url, config_file)
                except Exception as exc:
                    raise FileNotFoundError(f"Downloading standard config for {key} failed, possible solutions:\n\
                          1. delete {config_file} and retry\n\
                          2. manually download {url} to {config_file}") from exc
            config = load_config(config_file)

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
        return loss

    def save_model(self, ckpt_path=None):
        """save model"""

        if ckpt_path is not None:
            ms.save_checkpoint(self.model.network, ckpt_path)
        else:
            print("Checkpoint path is None!")
