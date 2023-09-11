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

"""problem structure"""
from abc import abstractmethod, ABC

from matplotlib import pyplot as plt
from sciai.utils import amp2datatype, datatype2np


class Problem(ABC):
    """Problem definition"""
    def __init__(self, args):
        self.dtype = amp2datatype(args.amp_level)
        self.dtype_np = datatype2np(self.dtype)
        self.epochs = args.epochs

        self.num_domain = args.num_domain
        self.num_initial = args.num_initial
        self.x_range = args.x_range
        self.t_range = args.t_range
        self.num_boundary = args.num_boundary

        self.save_fig = args.save_fig
        self.figures_path = args.figures_path

        self.loss = []
        self.steps = []

    @abstractmethod
    def setup_train_cell(self, args, net):
        pass

    @abstractmethod
    def setup_networks(self, args):
        pass

    @abstractmethod
    def train(self, train_cell):
        pass

    @abstractmethod
    def predict(self, network, *inputs):
        pass

    @abstractmethod
    def func(self, x, t):
        pass

    @abstractmethod
    def plot_result(self, x_test, t_test, y_test, y_res):
        pass

    @abstractmethod
    def generate_data(self, num):
        pass

    def plot_train_process(self):
        plt.figure(1)
        plt.semilogy(self.steps, self.loss, label="Train loss")
        plt.xlabel("# Steps")
        plt.legend()
        plt.savefig(f"{self.figures_path}/loss_history_ms.png")
