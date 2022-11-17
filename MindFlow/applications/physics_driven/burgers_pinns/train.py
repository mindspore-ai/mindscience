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
# ============================================================================
"""train process"""
import os
import time
import numpy as np

from mindspore.common import set_seed
from mindspore import context, nn
from mindspore.train import DynamicLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype

from mindflow.loss import Constraints
from mindflow.solver import Solver
from mindflow.common import LossAndTimeMonitor
from mindflow.cell import FCSequential
from mindflow.pde import Burgers1D
from mindflow.utils import load_yaml_config

from src.dataset import create_random_dataset
from src import visual_result

set_seed(123456)
np.random.seed(123456)

context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU", device_id=6)


def train():
    """training process"""
    # load configurations
    config = load_yaml_config('burgers_cfg.yaml')

    # create dataset
    burgers_train_dataset = create_random_dataset(config)
    train_dataset = burgers_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                         shuffle=True,
                                                         prebatched_data=True,
                                                         drop_remainder=True)
    # define models and optimizers
    model = FCSequential(in_channels=config["model"]["in_channels"],
                         out_channels=config["model"]["out_channels"],
                         layers=config["model"]["layers"],
                         neurons=config["model"]["neurons"],
                         residual=config["model"]["residual"],
                         act=config["model"]["activation"])
    if config["load_ckpt"]:
        param_dict = load_checkpoint(config["load_ckpt_path"])
        load_param_into_net(model, param_dict)
    if context.get_context(attr_key='device_target') == "Ascend":
        model.to_float(mstype.float16)
    optimizer = nn.Adam(model.trainable_params(), config["optimizer"]["initial_lr"])

    # define constraints
    burgers_problems = [Burgers1D(model=model) for _ in range(burgers_train_dataset.num_dataset)]
    train_constraints = Constraints(burgers_train_dataset, burgers_problems)

    # define solvers
    solver = Solver(model,
                    optimizer=optimizer,
                    train_constraints=train_constraints,
                    loss_scale_manager=DynamicLossScaleManager(),
                    )
    # define callbacks
    callbacks = [LossAndTimeMonitor(len(burgers_train_dataset))]
    if config["save_ckpt"]:
        config_ck = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=2)
        ckpoint_cb = ModelCheckpoint(prefix='burgers_1d', directory=config["save_ckpt_path"], config=config_ck)
        callbacks += [ckpoint_cb]

    # run the solver to train the model with callbacks
    solver.train(config["train_epochs"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)
    # visualization
    visual_result(model, resolution=config["visual_resolution"])


if __name__ == '__main__':
    print("pid:", os.getpid())
    time_beg = time.time()
    train()
    print("End-to-End total time: {} s".format(time.time() - time_beg))
