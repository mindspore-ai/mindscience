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
"""train process"""

import argparse
import os
import time

import numpy as np

from mindspore import context, nn, get_seed, set_seed, data_sink

from mindflow.cell import MultiScaleFCSequential
from mindflow.utils import load_yaml_config
from mindflow.utils import print_log, log_config, log_timer

from src import create_train_dataset, create_problem, create_trainer, create_normal_params
from src import re_initialize_model, evaluate, plot_l2_comparison_error
from src import WorkspaceConfig

set_seed(0)
np.random.seed(0)


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser(description="meta-pinns")
    parser.add_argument("--case", type=str, default="burgers",
                        choices=["burgers", "l_burgers",
                                 "cylinder_flow", "periodic_hill"],
                        help="choose burgers")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/burgers.yaml")
    result_args = parser.parse_args()
    return result_args


@log_timer
def train(args):
    '''Train and evaluate the network'''
    # load configurations
    case_name = args.case
    config = load_yaml_config(args.config_file_path)
    model_config = config["model"]
    test_config = config["meta_test"]
    summary_config = config["summary"]
    lamda_config = config["lamda"]
    meta_train_config = config["meta_train"]
    initial_lr = config["optimizer"]["initial_lr"]

    # create dataset
    inner_train_dataset = create_train_dataset(
        case_name, config, get_seed() + 1)
    outer_train_dataset = create_train_dataset(
        case_name, config, get_seed() + 2)

    # define models and optimizers
    model = MultiScaleFCSequential(in_channels=model_config["in_channels"],
                                   out_channels=model_config["out_channels"],
                                   layers=model_config["layers"],
                                   neurons=model_config["neurons"],
                                   residual=model_config["residual"],
                                   act=model_config["activation"],
                                   num_scales=1)

    lamda = lamda_config["initial_lamda"]
    problem = create_problem(lamda, case_name, model, config)
    inner_optimizer = nn.SGD(model.trainable_params(),
                             initial_lr)
    outer_optimizer = nn.Adam(problem.get_params(),
                              initial_lr)

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, model_config["amp_level"])
    else:
        loss_scaler = None

    inner_trainer = create_trainer(case_name, model, inner_optimizer, problem)
    inner_trainer.set_params(use_ascend, loss_scaler, False, True)
    outer_trainer = create_trainer(case_name, model, outer_optimizer, problem)
    outer_trainer.set_params(use_ascend, loss_scaler, True, True)
    inner_train_step = inner_trainer.train_step
    outer_train_step = outer_trainer.train_step

    iteration_str = "iterations"
    inner_iters = meta_train_config["inner_loop"][iteration_str]
    outer_iters = meta_train_config["outer_loop"][iteration_str]

    steps_per_epochs = inner_train_dataset.get_dataset_size()
    inner_sink_process = data_sink(
        inner_train_step, inner_train_dataset, sink_size=1)
    outer_sink_process = data_sink(
        outer_train_step, outer_train_dataset, sink_size=1)

    lamda_min = lamda_config["lamda_min"]
    lamda_max = lamda_config["lamda_max"]

    used_lamda = [lamda_config["eva_lamda"]]
    best_params = problem.get_params()
    best_l2 = 1e10

    # starting meta training
    eva_l2_errors = []
    for epoch in range(1, 1 + outer_iters):
        # train
        lamda = np.random.uniform(lamda_min, lamda_max)
        if lamda not in used_lamda:
            used_lamda.append(lamda)
        time_beg = time.time()
        model.set_train(True)

        if epoch % meta_train_config["reinit_lamda"] == 0:
            problem.lamda = lamda
        if epoch % meta_train_config["reinit_epoch"] == 0:
            re_initialize_model(model, epoch)

        for _ in range(1, 1 + inner_iters):
            for _ in range(steps_per_epochs):
                inner_sink_process()
        for _ in range(steps_per_epochs):
            cur_loss = outer_sink_process()

        print_log(
            f"epoch: {epoch} loss: {cur_loss} epoch time: {(time.time() - time_beg) * 1000}ms")

        if epoch % meta_train_config["eva_interval_outer"] == 0:
            # evaluate current model on unseen lamda
            print_log(
                f"learned params are: {problem.get_params(if_value=True)}")
            eva_iter = meta_train_config["eva_loop"][iteration_str]
            eva_l2_error = evaluate(WorkspaceConfig(epoch, case_name, config, problem.get_params(), eva_iter,
                                                    eva_iter, use_ascend, loss_scaler, False, True, False, None))
            eva_l2_errors.append(eva_l2_error[0])
            if eva_l2_error[0] < best_l2:
                best_l2 = eva_l2_error[0]
                best_params = problem.get_params()

    print_log(best_l2)
    for param in best_params:
        print_log(param.asnumpy())

    # start comparing
    test_iter = test_config[iteration_str]
    test_interval = test_config["cal_l2_interval"]

    # start meta training
    meta_l2_errors = evaluate(WorkspaceConfig(None, case_name, config, best_params,
                                              test_iter, test_interval,
                                              use_ascend, loss_scaler, False, True,
                                              True, f"{case_name}_meta_testing"))
    # end meta training

    # start normal training
    normal_params = create_normal_params(case_name)

    normal_l2_errors = evaluate(WorkspaceConfig(None, case_name, config, normal_params,
                                                test_iter, test_interval,
                                                use_ascend, loss_scaler, False, False,
                                                True, f"{case_name}_normal_training"))
    # end normal training

    plot_l2_comparison_error(
        case_name, summary_config["visual_dir"], test_interval, meta_l2_errors, normal_l2_errors)


if __name__ == '__main__':
    input_args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if input_args.mode.upper().startswith("GRAPH")
                        else context.PYNATIVE_MODE,
                        device_target=input_args.device_target,
                        device_id=input_args.device_id)
    log_config('./logs', f'{input_args.case}')
    print_log(
        f"Running in {input_args.mode.upper()} mode, using device id: {input_args.device_id}.")
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(use_ascend)
    print_log(f"pid: {os.getpid()}")
    start_time = time.time()
    train(input_args)
    print_log(f"End-to-End total time: {time.time() - start_time} s")
