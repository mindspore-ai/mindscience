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
"""dataset"""
from abc import ABCMeta, abstractmethod


def curry1(f):
    """Supply all arguments but the first."""

    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def data_process_run(data, funcs):
    for f in funcs:
        data = f(data)
    return data


class DataSet(metaclass=ABCMeta):
    """DataSet"""
    def __init__(self):
        self.phase = None

    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def set_phase(self, phase):
        self.phase = phase

    @abstractmethod
    def process(self, data, **kwargs):
        pass

    @abstractmethod
    def download(self, path=None):
        pass

    @abstractmethod
    def data_parse(self, idx):
        pass

    @abstractmethod
    def create_iterator(self, num_epochs, **kwargs):
        pass
