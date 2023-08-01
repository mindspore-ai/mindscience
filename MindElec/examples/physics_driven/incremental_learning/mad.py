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
"""reconstruct process."""
import argparse
import json
import math
import os
import time

import numpy as np
import mindspore as ms
from mindspore import context, Tensor, nn, Parameter, set_seed
from mindspore.common.initializer import HeUniform
from mindspore.train import DynamicLossScaleManager, ModelCheckpoint, CheckpointConfig, load_checkpoint, \
    load_param_into_net

from mindelec.architecture import MultiScaleFCCell, MTLWeightedLossCell
from mindelec.common import L2
from mindelec.loss import Constraints
from mindelec.solver import Solver, LossAndTimeMonitor
from src import get_test_data, create_random_dataset, MultiStepLR, Maxwell2DMur, PredictCallback, visual_result


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Meta Auto Decoder Simulation')
    parser.add_argument('--mode', type=str, default="pretrain", choices=["pretrain", "reconstruct"],
                        help="Running mode options: pretrain or reconstruct")
    opt = parser.parse_args()
    return opt


def piad2d(args):
    """pretraining and reconstruction process"""
    config = preprocess_config(args)
    train_dataset = create_random_dataset(config)
    train_dataset = train_dataset.create_dataset(batch_size=config["batch_size"], shuffle=True,
                                                 prebatched_data=True, drop_remainder=True)
    epoch_steps = len(train_dataset)
    print("check train dataset size: ", len(train_dataset))
    # load ckpt
    if config.get("load_ckpt", False):
        param_dict = load_checkpoint(config["load_ckpt_path"])
        if args.mode == "pretrain":
            loaded_ckpt_dict = param_dict
        else:
            loaded_ckpt_dict, latent_vector_ckpt = {}, 0
            for name in param_dict:
                if name == "model.latent_vector":
                    latent_vector_ckpt = param_dict[name].data.asnumpy()
                elif "network" in name and "moment" not in name:
                    loaded_ckpt_dict[name] = param_dict[name]
    # initialize latent vector
    num_scenarios, latent_size = config["num_scenarios"], config["latent_vector_size"]
    latent_vector = calc_latent_init(latent_size, latent_vector_ckpt, args.mode, num_scenarios)
    network = MultiScaleFCCell(config["input_size"], config["output_size"],
                               layers=config["layers"], neurons=config["neurons"], residual=config["residual"],
                               weight_init=HeUniform(negative_slope=math.sqrt(5)), act="sin",
                               num_scales=config["num_scales"], amp_factor=config["amp_factor"],
                               scale_factor=config["scale_factor"], input_scale=config["input_scale"],
                               input_center=config["input_center"], latent_vector=latent_vector)
    network = network.to_float(ms.float16)
    network.input_scale.to_float(ms.float32)
    mtl_cell = MTLWeightedLossCell(num_losses=train_dataset.num_dataset) if config.get("enable_mtl", True) else None
    # define problem
    train_prob = {}
    for dataset in train_dataset.all_datasets:
        train_prob[dataset.name] = Maxwell2DMur(network=network, config=config, domain_column=dataset.name + "_points",
                                                ic_column=dataset.name + "_points", bc_column=dataset.name + "_points")
    print("check problem: ", train_prob)
    train_constraints = Constraints(train_dataset, train_prob)
    # optimizer
    params = load_net(args, config, loaded_ckpt_dict, mtl_cell, network)
    lr_scheduler = MultiStepLR(config["lr"], config["milestones"], config["lr_gamma"], epoch_steps,
                               config["train_epoch"])
    optimizer = nn.Adam(params, learning_rate=Tensor(lr_scheduler.get_lr()))
    # problem solver
    solver = Solver(network, optimizer=optimizer, mode="PINNs", train_constraints=train_constraints,
                    test_constraints=None, metrics={'l2': L2(), 'distance': nn.MAE()}, loss_fn='smooth_l1_loss',
                    loss_scale_manager=DynamicLossScaleManager(), mtl_weighted_cell=mtl_cell,
                    latent_vector=latent_vector, latent_reg=config["latent_reg"])
    callbacks = get_callbacks(args, config, epoch_steps, network)
    solver.train(config["train_epoch"], train_dataset, callbacks=callbacks, dataset_sink_mode=True)


def load_net(args, config, loaded_ckpt_dict, mtl_cell, network):
    """load params into net"""
    if args.mode == "pretrain":
        params = network.trainable_params() + mtl_cell.trainable_params()
        if config.get("load_ckpt", False):
            load_param_into_net(network, loaded_ckpt_dict)
            load_param_into_net(mtl_cell, loaded_ckpt_dict)
    else:
        if config.get("finetune_model"):
            model_params = network.trainable_params()
        else:
            model_params = [param for param in network.trainable_params()
                            if ("bias" not in param.name and "weight" not in param.name)]
        params = model_params + mtl_cell.trainable_params() if mtl_cell else model_params
        load_param_into_net(network, loaded_ckpt_dict)
    return params


def load_ckpt(args, config):
    """load checkpoint into dict"""
    if config.get("load_ckpt", False):
        param_dict = load_checkpoint(config["load_ckpt_path"])
        if args.mode == "pretrain":
            loaded_ckpt_dict = param_dict
            latent_vector_ckpt = 0
        else:
            loaded_ckpt_dict = {}
            latent_vector_ckpt = 0
            for name in param_dict:
                if name == "model.latent_vector":
                    latent_vector_ckpt = param_dict[name].data.asnumpy()
                elif "network" in name and "moment" not in name:
                    loaded_ckpt_dict[name] = param_dict[name]
    return latent_vector_ckpt, loaded_ckpt_dict


def get_callbacks(args, config, epoch_steps, network):
    """get callbacks"""
    callbacks = [LossAndTimeMonitor(epoch_steps)]
    if config.get("train_with_eval", False):
        input_data, label_data = get_test_data(config["test_data_path"])
        eval_callback = PredictCallback(network, input_data, label_data, config=config, visual_fn=visual_result)
        callbacks += [eval_callback]
    if config["save_ckpt"]:
        config_ck = CheckpointConfig(save_checkpoint_steps=10, keep_checkpoint_max=2)
        prefix = 'pretrain_maxwell_frq1e9' if args.mode == "pretrain" else 'reconstruct_maxwell_frq1e9'
        ckpoint_cb = ModelCheckpoint(prefix=prefix, directory=config["save_ckpt_path"], config=config_ck)
        callbacks += [ckpoint_cb]
    return callbacks


def calc_latent_init(latent_size, latent_vector_ckpt, mode, num_scenarios):
    if mode == "pretrain":
        latent_init = np.random.randn(num_scenarios, latent_size) / np.sqrt(latent_size)
    else:
        latent_norm = np.mean(np.linalg.norm(latent_vector_ckpt, axis=1))
        print("check mean latent vector norm: ", latent_norm)
        latent_init = np.zeros((num_scenarios, latent_size))
    latent_vector = Parameter(Tensor(latent_init, ms.float32), requires_grad=True)
    return latent_vector


def preprocess_config(args):
    """preprocess to get the coefficients of electromagnetic field for each scenario"""
    if args.mode == "pretrain":
        config = json.load(open("./config/pretrain.json"))
    else:
        config = json.load(open("./config/reconstruct.json"))
    eps_candidates = config["EPS_candidates"]
    mu_candidates = config["MU_candidates"]
    config["num_scenarios"] = len(eps_candidates) * len(mu_candidates)
    batch_size_single_scenario = config["train_batch_size"]
    config["batch_size"] = batch_size_single_scenario * config["num_scenarios"]
    eps_list = []
    for eps in eps_candidates:
        eps_list.extend([eps] * (batch_size_single_scenario * len(mu_candidates)))
    mu_list = []
    for mu in mu_candidates:
        mu_list.extend([mu] * batch_size_single_scenario)
    mu_list = mu_list * (len(eps_candidates))

    exp_name = "_" + config["Case"] + '_num_scenarios_' + str(config["num_scenarios"]) \
               + "_latent_reg_" + str(config["latent_reg"])
    if config["save_ckpt"]:
        config["save_ckpt_path"] += exp_name

    config["vision_path"] += exp_name
    config["summary_path"] += exp_name
    print("check config: {}".format(config))
    config["eps_list"] = eps_list
    config["mu_list"] = mu_list
    config["load_ckpt"] = True if args.mode == "reconstruct" else config.get("load_ckpt", False)
    return config


if __name__ == '__main__':
    print("pid:", os.getpid())
    time_beg = time.time()
    set_seed(123456)
    np.random.seed(123456)
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", save_graphs_path="./solver")
    opts = parse_args()
    piad2d(opts)
    print("End-to-End total time: {} s".format(time.time() - time_beg))
