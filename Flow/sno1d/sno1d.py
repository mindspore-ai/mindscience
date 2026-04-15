# Copyright 2024 Huawei Technologies Co., Ltd
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
"""train"""
import os
import time
import argparse
import datetime

import numpy as np
from mindspore import context, nn, Tensor, set_seed, ops, data_sink, jit, load_checkpoint, load_param_into_net
from mindspore import dtype as mstype
from mindflow import SNO1D, get_poly_transform, RelativeRMSELoss, load_yaml_config, get_warmup_cosine_annealing_lr
from mindflow.pde import UnsteadyFlowWithLoss
from mindflow.utils import log_config, print_log, log_timer

from src import create_training_dataset, load_interp_data, test_error, visual


set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Burgers 1D problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend","CPU"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=3, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./configs/sno1d.yaml")
    input_args = parser.parse_args()
    return input_args

def test(input_args):
    '''evaluate the network'''
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]

    # prepare dataset
    poly_type = data_params['poly_type']
    load_interp_data(data_params, dataset_type='train')

    test_data = load_interp_data(data_params, dataset_type='test')
    test_input = Tensor(test_data['test_inputs'], mstype.float32)
    test_label = Tensor(test_data['test_labels'], mstype.float32)

    data_path = data_params['root_dir']
    label_unif = np.load(os.path.join(data_path, "test/label.npy"))

    # prepare model
    n_modes = model_params['modes']
    resolution = data_params['resolution']

    transform_data = get_poly_transform(resolution, n_modes, poly_type)

    transform = Tensor(transform_data["analysis"], mstype.float32)
    inv_transform = Tensor(transform_data["synthesis"], mstype.float32)

    model = SNO1D(
        in_channels=model_params['in_channels'],
        out_channels=model_params['out_channels'],
        hidden_channels=model_params['hidden_channels'],
        num_sno_layers=model_params['sno_layers'],
        transforms=[[transform, inv_transform]],
        compute_dtype=mstype.float32)

    param_dict = load_checkpoint(config["summary"]["ckpt_dir"])
    load_param_into_net(model, param_dict)
    model.set_train(False)
    test_error(model, test_input, test_label, label_unif, data_params)
    
if __name__ == '__main__':
    log_config("./logs", "sno1d")
    print_log(f"pid: {os.getpid()}")
    print_log(datetime.datetime.now())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    test(args)
