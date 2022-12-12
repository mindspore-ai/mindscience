# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
train
"""
import argparse
import os

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager

from mindflow.cell.neural_operators import FNO1D
from mindflow.solver import Solver
from src.callback import PredictCallback
from src.dataset import create_dataset
from src.loss import RelativeRMSELoss
from src.lr_scheduler import warmup_cosine_annealing_lr
from src.utils import load_config

set_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Burgers 1D problem')
parser.add_argument('--device_id', type=int, default=7)
parser.add_argument('--config_path', default='burgers1d.yaml',
                    help='yaml config file path')
opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target="GPU",
                    device_id=opt.device_id)


def train_with_eval():
    '''Train and evaluate the network'''
    config = load_config(opt.config_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    callback_params = config["callback"]

    train_dataset = create_dataset(data_params,
                                   shuffle=True)

    test_input, test_label = np.load(os.path.join(data_params["path"], "test/inputs.npy")), \
        np.load(os.path.join(data_params["path"], "test/label.npy"))

    model = FNO1D(in_channels=model_params["input_dims"],
                  out_channels=model_params["output_dims"],
                  resolution=model_params["resolution"],
                  modes=model_params["modes"],
                  channels=model_params["width"],
                  depths=model_params["depth"])
    model_params_list = []
    for k, v in model_params.items():
        model_params_list.append(f"{k}:{v}")
    model_name = "_".join(model_params_list)
    print(model_name)

    steps_per_epoch = train_dataset.get_dataset_size()
    lr = warmup_cosine_annealing_lr(lr=optimizer_params["initial_lr"],
                                    steps_per_epoch=steps_per_epoch,
                                    warmup_epochs=1,
                                    max_epoch=optimizer_params["train_epochs"])
    optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))
    loss_scale = DynamicLossScaleManager()

    loss_fn = RelativeRMSELoss()
    solver = Solver(model,
                    optimizer=optimizer,
                    loss_scale_manager=loss_scale,
                    loss_fn=loss_fn,
                    )

    summary_dir = os.path.join(callback_params["summary_dir"], model_name)
    print(summary_dir)
    pred_cb = PredictCallback(model=model,
                              inputs=test_input,
                              label=test_label,
                              config=config,
                              summary_dir=summary_dir)
    solver.train(epoch=optimizer_params["train_epochs"],
                 train_dataset=train_dataset,
                 callbacks=[LossMonitor(), TimeMonitor(), pred_cb],
                 dataset_sink_mode=True)


if __name__ == '__main__':
    train_with_eval()
