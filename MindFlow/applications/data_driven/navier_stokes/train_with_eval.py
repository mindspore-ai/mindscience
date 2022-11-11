# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import argparse
import datetime
import numpy as np

import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore import Tensor, context
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.train.loss_scale_manager import DynamicLossScaleManager

from mindflow.cell.neural_operators import FNO2D
from mindflow.solver import Solver

from src.callback import PredictCallback
from src.lr_scheduler import warmup_cosine_annealing_lr
from src.dataset import create_dataset
from src.utils import load_config
from src.loss import RelativeRMSELoss

set_seed(0)
np.random.seed(0)

print("pid:", os.getpid())
print(datetime.datetime.now())

parser = argparse.ArgumentParser(description='navier_stoke 2D problem')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--config_path', default='navier_stokes_2d.yaml', help='yaml config file path')
opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target='Ascend',
                    device_id=opt.device_id,
                    )


def train_with_eval():
    '''train and evaluate the network'''
    config = load_config(opt.config_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    callback_params = config["callback"]

    # prepare dataset
    train_dataset = create_dataset(data_params,
                                   input_resolution=model_params["input_resolution"],
                                   shuffle=True)
    test_input = np.load(os.path.join(data_params["path"], "test/inputs.npy"))
    test_label = np.load(os.path.join(data_params["path"], "test/label.npy"))

    # prepare model
    model = FNO2D(input_dims=model_params["input_dims"],
                  output_dims=model_params["output_dims"],
                  resolution=model_params["input_resolution"],
                  modes=model_params["modes"],
                  channels=model_params["width"],
                  depth=model_params["depth"]
                  )

    model_params_list = []
    for k, v in model_params.items():
        model_params_list.append(f"{k}-{v}")
    model_name = "_".join(model_params_list)

    # prepare optimizer
    steps_per_epoch = train_dataset.get_dataset_size()
    lr = warmup_cosine_annealing_lr(lr=optimizer_params["initial_lr"],
                                    steps_per_epoch=steps_per_epoch,
                                    warmup_epochs=optimizer_params["warmup_epochs"],
                                    max_epoch=optimizer_params["train_epochs"])

    optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))
    loss_scale = DynamicLossScaleManager()

    # prepare loss function
    loss_fn = RelativeRMSELoss()
    solver = Solver(model,
                    optimizer=optimizer,
                    loss_scale_manager=loss_scale,
                    loss_fn=loss_fn,
                    )

    # prepare callback
    summary_dir = os.path.join(callback_params["summary_dir"], model_name)
    print(summary_dir)
    pred_cb = PredictCallback(model=model,
                              inputs=test_input,
                              label=test_label,
                              config=callback_params,
                              summary_dir=summary_dir)

    ckpt_config = CheckpointConfig(save_checkpoint_steps=callback_params["save_checkpoint_steps"] * steps_per_epoch,
                                   keep_checkpoint_max=callback_params["keep_checkpoint_max"])
    ckpt_dir = os.path.join(summary_dir, "ckpt")
    ckpt_cb = ModelCheckpoint(prefix=model_params["name"],
                              directory=ckpt_dir,
                              config=ckpt_config)

    # start train with evaluation
    solver.train(epoch=optimizer_params["train_epochs"],
                 train_dataset=train_dataset,
                 callbacks=[LossMonitor(), TimeMonitor(), pred_cb, ckpt_cb],
                 dataset_sink_mode=True)


if __name__ == '__main__':
    train_with_eval()