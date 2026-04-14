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
# ==============================================================================
"""
train
"""
import os
import time
import datetime
import argparse
import numpy as np

from mindspore import nn, context, ops, Tensor, set_seed, dtype, data_sink, load_checkpoint, load_param_into_net
from mindspore.nn.loss import MSELoss

from mindflow.cell import KNO2D
from mindflow.core import get_warmup_cosine_annealing_lr
from mindflow.utils import load_yaml_config, print_log, log_config, log_timer

from src import create_training_dataset, NavierStokesWithLoss, visual

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Navier Stokes problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend", "CPU"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=1, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./configs/kno2d.yaml")
    input_args = parser.parse_args()
    return input_args


def test(input_args):
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]

    # create training dataset
    test_input = np.load(os.path.join(data_params["root_dir"], "test/inputs.npy"))
    test_label = np.load(os.path.join(data_params["root_dir"], "test/label.npy"))

    model = KNO2D(in_channels=data_params['in_channels'],
                  channels=model_params['channels'],
                  modes=model_params['modes'],
                  depths=model_params['depths'],
                  resolution=model_params['resolution'],
                  compute_dtype=dtype.float16 if use_ascend else dtype.float32
                  )

    loss_fn = MSELoss()
    problem = NavierStokesWithLoss(model, data_params["out_channels"], loss_fn, data_format="NHWTC")
    param_dict = load_checkpoint(config["summary"]["ckpt_dir"])
    load_param_into_net(model, param_dict)
    model.set_train(False)
    print_log("================================Start Evaluation================================")
    eval_time_start = time.time()
    l_recons_all, l_pred_all = problem.test(test_input, test_label)
    print_log(f'recons loss: {l_recons_all},'
            f' relative pred loss: {l_pred_all}')
    print_log("=================================End Evaluation=================================")
    print_log(f'evaluation time: {time.time() - eval_time_start}s')


if __name__ == '__main__':
    log_config('./logs', 'kno2d')
    print_log("pid:", os.getpid())
    print_log(datetime.datetime.now())
    args = parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    test(args)
