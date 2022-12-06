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

from mindflow.loss import Constraints
from mindflow.solver import Solver
from mindflow.common import L2
from mindflow.utils import load_yaml_config
from mindflow.common import LossAndTimeMonitor
from mindflow.cell import FCSequential

from src import create_random_dataset
from src import Darcy2D
from src import visual_result

set_seed(123456)
np.random.seed(123456)

context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="GPU",
    save_graphs_path="./graph",
)

def train(config):
    """training process"""
    geom_name = "flow_region"
    # create train dataset
    flow_train_dataset = create_random_dataset(config, geom_name)
    train_data = flow_train_dataset.create_dataset(
        batch_size=config["train_batch_size"], shuffle=True, drop_remainder=True
    )

    # network model
    model = FCSequential(
        in_channels=config["model"]["input_size"],
        out_channels=config["model"]["output_size"],
        neurons=config["model"]["neurons"],
        layers=config["model"]["layers"],
        residual=config["model"]["residual"],
        act=config["model"]["activation"],
        weight_init=config["model"]["weight_init"]
    )

    # define problem and Constraints
    darcy_problem = [
        Darcy2D(model=model) for _ in range(flow_train_dataset.num_dataset)
    ]
    train_constraints = Constraints(flow_train_dataset, darcy_problem)

    # optimizer
    params = model.trainable_params()
    optim = nn.Adam(params, learning_rate=config["optimizer"]["lr"])

    # solver
    solver = Solver(
        model,
        optimizer=optim,
        mode="PINNs",
        train_constraints=train_constraints,
        test_constraints=None,
        metrics={"l2": L2(), "distance": nn.MAE()},
        loss_scale_manager=DynamicLossScaleManager(),
    )

    # training

    # define callbacks
    callbacks = [LossAndTimeMonitor(len(flow_train_dataset))]

    if config["save_ckpt"]:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=2)
        ckpoint_cb = ModelCheckpoint(
            prefix="ckpt_darcy", directory=config["save_ckpt_path"], config=ckpt_config
        )
        callbacks += [ckpoint_cb]

    solver.train(
        epoch=config["train_epoch"], train_dataset=train_data, callbacks=callbacks
    )

    visual_result(model, config)

if __name__ == "__main__":
    print("pid:", os.getpid())
    configs = load_yaml_config("darcy_cfg.yaml")
    time_beg = time.time()
    train(configs)
    print(f"End-to-End total time: {format(time.time() - time_beg)} s")
