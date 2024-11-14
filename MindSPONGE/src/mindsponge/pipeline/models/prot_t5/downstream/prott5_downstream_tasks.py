# Copyright 2024 Huawei Technologies Co., Ltd
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
"""ProtT5 Downstream Task class implement."""
from mindspore import jit

from ...model import Model

from .hhblits_task import HHblitsTask
from .deeploc_task import DeeplocTask


class ProtT5DownstreamTasks(Model):
    '''ProtT5DownstreamTasks'''

    def __init__(self, config):
        self.name = "ProtT5DownstreamTasks"
        self.mixed_precision = False

        self.config = config
        self.checkpoint_url = "https://download.mindspore.cn/mindscience/mindsponge/ProtT5/checkpoint/prot_t5_xl.ckpt"
        self.checkpoint_path = "./prot_t5_xl.ckpt"

        self.mode = config.mode
        self.task_name = config.task_name

        if self.task_name == "hhblits":
            self.network = HHblitsTask(config)
        elif self.task_name == "deeploc":
            self.network = DeeplocTask(config)
        else:
            raise ValueError(f"Undefined task: {self.task_name}")

        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network, self.name, None,
                         mixed_precision=self.mixed_precision)


    def forward(self, data):
        pass

    def backward(self, data):
        pass

    # pylint: disable=W0221
    def predict(self, data):
        return self.network.predict(data)

    def eval_acc(self, data_path):
        """eval accuracy of data file"""
        if self.task_name == "hhblits":
            m3acc, m8acc = self.network.eval_acc(data_path)
            print("Accuracy Q3 %.4f; Q8 %.4f" % (m3acc, m8acc))
        elif self.task_name == "deeploc":
            acc = self.network.eval_acc(data_path)
            print("Accuracy %.4f" % acc)

    def train(self):
        """train"""
        self.network.train()

    @jit
    def train_step(self, data):
        self.network.train_step(*data)

    @jit
    def _jit_forward(self, data):
        pass

    def _pynative_forward(self, data):
        pass
