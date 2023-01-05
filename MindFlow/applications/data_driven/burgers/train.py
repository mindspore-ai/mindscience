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

from mindspore import context, nn, Tensor, set_seed
from mindspore import DynamicLossScaleManager, LossMonitor, TimeMonitor

from mindflow import FNO1D, RelativeRMSELoss, Solver, load_yaml_config, get_warmup_cosine_annealing_lr

from src import PredictCallback, create_training_dataset


set_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Burgers 1D problem')
parser.add_argument('--device_id', type=int, default=4)
parser.add_argument('--config_path', default='burgers1d.yaml', help='yaml config file path')
opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target="GPU",
                    device_id=opt.device_id)


def train():
    '''Train and evaluate the network'''
    config = load_yaml_config(opt.config_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    callback_params = config["callback"]

    # create training dataset
    train_dataset = create_training_dataset(data_params, shuffle=True)

    # create test dataset
    test_input, test_label = np.load(os.path.join(data_params["path"], "test/inputs.npy")), \
                             np.load(os.path.join(data_params["path"], "test/label.npy"))

    model = FNO1D(in_channels=model_params["in_channels"],
                  out_channels=model_params["out_channels"],
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
    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["initial_lr"],
                                        last_epoch=optimizer_params["train_epochs"],
                                        steps_per_epoch=steps_per_epoch,
                                        warmup_epochs=1)
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
    train()
