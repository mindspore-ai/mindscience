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
import json
import time
import numpy as np

from mindspore.common import set_seed
from mindspore import context, nn
from mindspore.train import DynamicLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype

from mindflow.loss import Constraints
from mindflow.solver import Solver, LossAndTimeMonitor
from mindflow.cell import FCSequential

from src import create_random_dataset
from src import Burgers1D
from src import visual_result

set_seed(123456)
np.random.seed(123456)

context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", device_id=0)


def train(config):
    """training process"""

    burgers_train_dataset = create_random_dataset(config)
    train_dataset = burgers_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                         shuffle=True,
                                                         prebatched_data=True,
                                                         drop_remainder=True)
    steps_per_epoch = len(burgers_train_dataset)
    print("check train dataset size: {}".format(len(burgers_train_dataset)))

    model = FCSequential(in_channel=2, out_channel=1, layers=6, neurons=20, residual=False, act="tanh")

    if context.get_context(attr_key='device_target') == "Ascend":
        model.to_float(mstype.float16)

    train_prob = {}
    for dataset in burgers_train_dataset.all_datasets:
        train_prob[dataset.name] = Burgers1D(model=model, config=config,
                                             domain_name="{}_points".format(dataset.name),
                                             ic_name="{}_points".format(dataset.name),
                                             bc_name="{}_points".format(dataset.name))
    print("check problem: ", train_prob)
    train_constraints = Constraints(burgers_train_dataset, train_prob)

    params = model.trainable_params()
    optim = nn.Adam(params, 5e-3)

    if config["load_ckpt"]:
        param_dict = load_checkpoint(config["load_ckpt_path"])
        load_param_into_net(model, param_dict)

    solver = Solver(model,
                    optimizer=optim,
                    train_constraints=train_constraints,
                    test_constraints=None,
                    loss_scale_manager=DynamicLossScaleManager(),
                    )

    loss_time_callback = LossAndTimeMonitor(steps_per_epoch)
    callbacks = [loss_time_callback]

    if config["save_ckpt"]:
        config_ck = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=2)
        ckpoint_cb = ModelCheckpoint(prefix='burgers_1d', directory=config["save_ckpt_path"], config=config_ck)
        callbacks += [ckpoint_cb]

    solver.train(config["train_epoch"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)

    visual_result(model, resolution=config["visual_resolution"])


if __name__ == '__main__':
    print("pid:", os.getpid())
    configs = json.load(open("./config.json"))
    print("check config: {}".format(configs))
    time_beg = time.time()
    train(configs)
    print("End-to-End total time: {} s".format(time.time() - time_beg))
