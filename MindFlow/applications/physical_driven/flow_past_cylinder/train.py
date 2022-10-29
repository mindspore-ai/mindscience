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
import os
import json
import time
import numpy as np

from mindspore.common import set_seed
from mindspore import context, Tensor, nn
from mindspore.train import DynamicLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype

from mindflow.loss import Constraints
from mindflow.solver import Solver, LossAndTimeMonitor
from mindflow.common import L2
from mindflow.loss import MTLWeightedLossCell

from src import create_training_dataset, create_evaluation_dataset
from src import NavierStokes2D, FlowNetwork
from src import MultiStepLR, PredictCallback, visualization

set_seed(123456)
np.random.seed(123456)

context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU", save_graphs_path="./graph",
                    device_id=0)


def train(config):
    """training process"""
    cylinder_dataset = create_training_dataset(config)
    train_dataset = cylinder_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                    shuffle=True,
                                                    prebatched_data=True,
                                                    drop_remainder=True)
    steps_per_epoch = len(cylinder_dataset)
    print("check training dataset size: ", steps_per_epoch)

    model = FlowNetwork(config["input_size"],
                        config["output_size"],
                        coord_min=config["coord_min"] + [0],
                        coord_max=config["coord_max"] + [config["range_t"]],
                        num_layers=config["layers"],
                        neurons=config["neurons"],
                        residual=config["residual"])
    if context.get_context(attr_key='device_target') == "Ascend":
        model.to_float(mstype.float16)
    mtl = MTLWeightedLossCell(num_losses=cylinder_dataset.num_dataset)
    print("Use MtlWeightedLossCell, num loss: {}".format(mtl.num_losses))

    train_prob = {}
    for dataset in cylinder_dataset.all_datasets:
        train_prob[dataset.name] = NavierStokes2D(model=model,
                                                  domain_points=dataset.name + "_points",
                                                  ic_points=dataset.name + "_points",
                                                  bc_points=dataset.name + "_points",
                                                  ic_label=dataset.name + "_label",
                                                  bc_label=dataset.name + "_label",
                                                  re=config["Reynolds_number"])
    print("check problem: ", train_prob)
    train_constraints = Constraints(cylinder_dataset, train_prob)

    params = model.trainable_params() + mtl.trainable_params()
    lr_scheduler = MultiStepLR(config["lr"], config["milestones"], config["lr_gamma"],
                               steps_per_epoch, config["train_epoch"])
    lr = lr_scheduler.get_lr()
    optim = nn.Adam(params, learning_rate=Tensor(lr))

    if config["load_ckpt"]:
        param_dict = load_checkpoint(config["load_ckpt_path"])
        load_param_into_net(model, param_dict)
        load_param_into_net(mtl, param_dict)

    solver = Solver(model,
                    optimizer=optim,
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
        inputs, label = create_evaluation_dataset(config["test_data_path"])
        predict_callback = PredictCallback(model, inputs, label, config=config, visual_fn=visualization)
        callbacks += [predict_callback]
    if config["save_ckpt"]:
        config_ck = CheckpointConfig(save_checkpoint_steps=10,
                                     keep_checkpoint_max=2)
        ckpoint_cb = ModelCheckpoint(prefix='ckpt_flow_past_cylinder_Re100',
                                     directory=config["save_ckpt_path"], config=config_ck)
        callbacks += [ckpoint_cb]

    solver.train(config["train_epoch"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)


if __name__ == '__main__':
    print("pid:", os.getpid())
    configs = json.load(open("./config.json"))
    print("check config: {}".format(configs))
    time_beg = time.time()
    train(configs)
    print("End-to-End total time: {} s".format(time.time() - time_beg))
