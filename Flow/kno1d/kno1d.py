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
test
"""
import os
import time
import argparse
import datetime
import numpy as np

from mindspore import nn, context, ops, Tensor, set_seed, data_sink, jit, load_checkpoint, load_param_into_net
from mindspore.nn.loss import MSELoss

from mindflow.cell import KNO1D
from mindflow.core import get_warmup_cosine_annealing_lr
from mindflow.utils import load_yaml_config, print_log, log_config, log_timer

from src import create_training_dataset, BurgersWithLoss, visual

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Burgers 1D problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend","CPU"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=2,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/kno1d.yaml")
    input_args = parser.parse_args()
    return input_args


@log_timer
def test(input_args):
    '''Train and evaluate the network'''
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]
    # create training dataset
    train_dataset = create_training_dataset(data_params, shuffle=True)

    # create test dataset
    eval_dataset = create_training_dataset(
        data_params, shuffle=False, is_train=False)

    model = KNO1D(in_channels=data_params['in_channels'],
                  channels=model_params['channels'],
                  modes=model_params['modes'],
                  depths=model_params['depths'],
                  resolution=model_params['resolution']
                  )

    eval_size = eval_dataset.get_dataset_size()
    loss_fn = MSELoss()
    problem = BurgersWithLoss(model, data_params["out_channels"], loss_fn)
    
    param_dict = load_checkpoint(config["summary"]["ckpt_dir"])
    load_param_into_net(model, param_dict)
    model.set_train(False)
    
    @jit
    def eval_step(inputs, labels):
        return problem.get_rel_loss(inputs, labels)
    
    l_recons_eval = 0.0
    l_pred_eval = 0.0

    for inputs, labels in eval_dataset.create_tuple_iterator():
        l_recons, l_pred = eval_step(inputs, labels)
        l_recons_eval += l_recons.asnumpy()
        l_pred_eval += l_pred.asnumpy()

    l_recons_eval /= eval_size
    l_pred_eval /= eval_size
    
    print_log(f'recons loss: {l_recons_eval},'
             f' relative pred loss: {l_pred_eval}')

if __name__ == '__main__':
    log_config('./logs', 'kno1d')
    print_log("pid:", os.getpid())
    print_log(datetime.datetime.now())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    test(args)
