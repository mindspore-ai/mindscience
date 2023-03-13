# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""Models"""
from abc import ABCMeta, abstractmethod
import os
import ssl
import urllib.request
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import load_checkpoint
from mindsponge.pipeline.cell.amp import amp_convert


class Model(metaclass=ABCMeta):
    """Model"""
    def __init__(self, checkpoint_url=None, network=None, name=None, white_list=None):
        self.cache = None
        self.ckpt_path = None
        self.checkpoint_url = checkpoint_url
        self.checkpoint_path = None
        self.name = name
        self.network = network
        self.white_list = white_list
        if ms.get_context("device_target") == "Ascend":
            self.network.to_float(mstype.float16)
            amp_convert(self.network, self.white_list)
        self._check_initialize()

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, data):
        pass

    @abstractmethod
    def train_step(self, data):
        pass

    @abstractmethod
    def predict(self, data, **kwargs):
        pass

    def set_cache(self, path):
        self.cache = path

    def set_checkpoint_path(self, path):
        self.ckpt_path = path

    def from_pretrained(self, ckpt_path=None):
        "from_pretrained"
        if ckpt_path is not None:
            self.checkpoint_path = ckpt_path
        if self.checkpoint_path is None:
            self.checkpoint_path = "./"
        if not os.path.exists(self.checkpoint_path):
            print("Download checkpoint to ", self.checkpoint_path)
            # pylint: disable=protected-access
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(self.checkpoint_url, self.checkpoint_path)
        load_checkpoint(self.checkpoint_path, self.network)

    def _check_initialize(self):
        if self.checkpoint_url is None:
            raise ValueError("checkpoint url is not initialize, please check your init function")
        if self.config is None:
            raise ValueError("model config is not initialize, please check your init function")
        if self.network is None:
            raise ValueError("network is not initialize, please check your init function")

    @abstractmethod
    def _jit_forward(self, data):
        pass

    @abstractmethod
    def _pynative_forward(self, data):
        pass
