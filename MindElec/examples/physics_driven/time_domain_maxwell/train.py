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
# ============================================================================
"""train process"""
import json
import math
import os
import time

import numpy as np
import mindspore as ms
from mindspore import context, Tensor, nn
from mindspore.common import set_seed
from mindspore.common.initializer import HeUniform
from mindspore.train import DynamicLossScaleManager, ModelCheckpoint, CheckpointConfig, load_checkpoint, \
    load_param_into_net

from mindelec.architecture import MultiScaleFCCell, MTLWeightedLossCell
from mindelec.common import L2
from mindelec.loss import Constraints
from mindelec.solver import Solver, LossAndTimeMonitor
from src import Maxwell2DMur, MultiStepLR, PredictCallback, create_train_dataset, get_test_data, \
    create_random_dataset, visual_result


def load_config():
    """load config"""
    with open(os.path.join(os.path.dirname(__file__), "config.json")) as f:
        config = json.load(f)
    return config


def train(config):
    """training process"""
    if config["random_sampling"]:
        elec_train_dataset = create_random_dataset(config)
    else:
        elec_train_dataset = create_train_dataset(config["train_data_path"])
    train_dataset = elec_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                      shuffle=True,
                                                      prebatched_data=True,
                                                      drop_remainder=True)
    steps_per_epoch = len(elec_train_dataset)
    print("check train dataset size: ", len(elec_train_dataset))

    # define network
    model = MultiScaleFCCell(config["input_size"],
                             config["output_size"],
                             layers=config["layers"],
                             neurons=config["neurons"],
                             input_scale=config["input_scale"],
                             residual=config["residual"],
                             weight_init=HeUniform(negative_slope=math.sqrt(5)),
                             act="sin",
                             num_scales=config["num_scales"],
                             amp_factor=config["amp_factor"],
                             scale_factor=config["scale_factor"]
                             )

    model.cell_list.to_float(ms.float16)
    mtl = MTLWeightedLossCell(num_losses=elec_train_dataset.num_dataset)

    # define problem
    train_prob = {}
    for dataset in elec_train_dataset.all_datasets:
        train_prob[dataset.name] = Maxwell2DMur(model=model, config=config,
                                                domain_name=dataset.name + "_points",
                                                ic_name=dataset.name + "_points",
                                                bc_name=dataset.name + "_points")
    print("check problem: ", train_prob)
    train_constraints = Constraints(elec_train_dataset, train_prob)

    # optimizer
    params = model.trainable_params() + mtl.trainable_params()
    lr_scheduler = MultiStepLR(config["lr"], config["milestones"], config["lr_gamma"],
                               steps_per_epoch, config["train_epoch"])
    lr = lr_scheduler.get_lr()
    optim = nn.Adam(params, learning_rate=Tensor(lr))

    if config["load_ckpt"]:
        param_dict = load_checkpoint(config["load_ckpt_path"])
        load_param_into_net(model, param_dict)
        load_param_into_net(mtl, param_dict)
    # define solver
    solver = Solver(model,
                    optimizer=optim,
                    mode="PINNs",
                    train_constraints=train_constraints,
                    test_constraints=None,
                    metrics={'l2': L2(), 'distance': nn.MAE()},
                    loss_fn='smooth_l1_loss',
                    loss_scale_manager=DynamicLossScaleManager(init_loss_scale=2 ** 10, scale_window=2000),
                    mtl_weighted_cell=mtl,
                    )

    loss_time_callback = LossAndTimeMonitor(steps_per_epoch)
    callbacks = [loss_time_callback]
    if config.get("train_with_eval", False):
        inputs, label = get_test_data(config["test_data_path"])
        predict_callback = PredictCallback(model, inputs, label, config=config, visual_fn=visual_result)
        callbacks += [predict_callback]
    if config["save_ckpt"]:
        config_ck = CheckpointConfig(save_checkpoint_steps=10,
                                     keep_checkpoint_max=2)
        ckpoint_cb = ModelCheckpoint(prefix='ckpt_maxwell_frq1e9',
                                     directory=config["save_ckpt_path"], config=config_ck)
        callbacks += [ckpoint_cb]

    solver.train(config["train_epoch"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)


if __name__ == '__main__':
    set_seed(123456)
    np.random.seed(123456)
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", save_graphs_path="./graph")
    print("pid:", os.getpid())
    configs = load_config()
    print("check config: {}".format(configs))
    time_beg = time.time()
    train(configs)
    print("End-to-End total time: {} s".format(time.time() - time_beg))
