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
"""train process of pinn network
"""
import os
import argparse
import random
import time

import numpy as np

from mindspore import nn, context, ops, set_seed
from mindspore import value_and_grad, jit, save_checkpoint
from mindspore import load_checkpoint, load_param_into_net
from mindspore.amp import DynamicLossScaler, auto_mixed_precision

from mindflow.utils import load_yaml_config
from mindflow.utils import print_log, log_config, log_timer
from mindflow.cell import MultiScaleFCSequential

from src import create_pinn_dataset, evaluate, cal_grads, cal_terms, pinn_loss_func
from src import get_dict_name, get_dicts, update_lib, calculate_coef, get_lefts

set_seed(0)
random.seed(0)


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser(description="train r_glda")
    parser.add_argument("--case", type=str, default="burgers", choices=["burgers", "cylinder_flow", "periodic_hill"],
                        help="choose burgers, cylinder_flow or periodic_hill")
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
    """train process"""
    # load configuration
    case_name = args.case

    config = load_yaml_config(args.config_file_path)
    pinn_config = config["pinn"]
    pinn_dataset_config = pinn_config["dataset"]
    model_config = config["model"]
    summary_config = config["summary"]
    optimizer_config = config["optimizer"]
    epochs = optimizer_config["epochs"]
    pinn_epochs = int(epochs / pinn_config["divide"])

    pinn_dataset = create_pinn_dataset(
        case_name, pinn_dataset_config)
    database_choose, h_data_choose, database_validate, h_data_validate = pinn_dataset

    model = MultiScaleFCSequential(in_channels=model_config["in_channels"],
                                   out_channels=model_config["out_channels"],
                                   layers=model_config["layers"],
                                   neurons=model_config["neurons"],
                                   residual=model_config["residual"],
                                   act=model_config["activation"],
                                   num_scales=1)

    # load checkpoint
    ckpt_name = f"{case_name}_nn-{epochs + 1}.ckpt"
    ckpt_path = summary_config["save_ckpt_path"]
    model_dict = load_checkpoint(os.path.join(
        ckpt_path, ckpt_name))
    load_param_into_net(model, model_dict)

    # Number the results obtained from the ga
    dict_name = get_dict_name(case_name)

    optimizer = nn.Adam(model.trainable_params(),
                        optimizer_config["initial_lr"])

    # set ascend
    if use_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, model_config["amp_level"])
    else:
        loss_scaler = None

    def forward_fn(dataset, lefts, coef_list, dict_name, terms_dict):
        database_choose, h_data_choose = dataset
        prediction = model(database_choose)
        f1 = nn.MSELoss(reduction='mean')(prediction, h_data_choose)
        loss = pinn_loss_func(f1, lefts, coef_list,
                              dict_name, terms_dict)

        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(dataset, lefts, coef_list, dict_name, terms_dict):
        loss, grads = grad_fn(dataset, lefts, coef_list, dict_name, terms_dict)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    print_log("----start training----")
    # train loop for pinn
    for epoch in range(1, pinn_epochs + 1):
        time_beg = time.time()
        prediction = model(database_choose)
        grads, libraries = cal_grads(case_name, model, database_choose,
                                     pinn_dataset_config["choose_train"])

        terms = cal_terms(case_name, prediction, grads)

        # terms in numpy
        terms_dict, dict_n = get_dicts(terms)

        libraries = update_lib(case_name, dict_name, libraries, dict_n)

        # Lasso
        lefts = get_lefts(case_name, grads, prediction, False)
        coef_list, lst_list = calculate_coef(lefts, libraries, epoch, config)

        model.set_train(True)

        # train step
        lefts = get_lefts(case_name, grads, prediction, True)

        step_train_loss = train_step((database_choose, h_data_choose),
                                     lefts, coef_list, dict_name, terms_dict)

        # set model to eval mode
        model.set_train(False)

        # put zeros
        if epoch >= 1000:
            for (j, lst) in enumerate(lst_list):
                for i in range(lst.shape[0]):
                    if np.abs(lst[i]) < pinn_config["kesi"]:
                        dict_name[j].pop(i)
                        break

        if epoch % summary_config["validate_interval_epochs"] == 0:
            print_log(f"Dict_Name {dict_name}")
            print_log(f"lst_list {lst_list}")
            # current epoch loss
            print_log(
                f"epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg) * 1000}ms")
            evaluate(model, database_validate, h_data_validate, config)

        # save checkpoint
        if epoch % summary_config["save_checkpoint_epochs"] == 0:
            ckpt_name = f"{case_name}_pinn-{epoch + 1}.ckpt"
            save_checkpoint(model, os.path.join(
                ckpt_path, ckpt_name))


if __name__ == '__main__':
    input_args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if input_args.mode.upper().startswith("GRAPH")
                        else context.PYNATIVE_MODE,
                        device_target=input_args.device_target,
                        device_id=input_args.device_id)
    log_config('./pinn_logs', f'{input_args.case}')
    print_log(
        f"Running in {input_args.mode.upper()} mode, using device id: {input_args.device_id}.")
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(use_ascend)
    print_log(f"pid: {os.getpid()}")
    start_time = time.time()
    train(input_args)
    print_log(f"End-to-End total time: {time.time() - start_time} s")
