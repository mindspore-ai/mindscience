# ============================================================================
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
"""inference process"""
import os
import argparse

import numpy as np

from mindspore import context
from mindspore import load_checkpoint, load_param_into_net

from mindflow.utils import load_yaml_config

from src import ResUnet3D, create_dataset, RRMSE, UnsteadyFlow3D, check_file_path, plot_results


def parse_args():
    """Parse input args"""
    parser = argparse.ArgumentParser(
        description='model eval for 3d unsteady flow')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./config.yaml")
    parser.add_argument("--norm", type=bool, default=False, choices=[True, False],
                        help="Whether to perform data normalization on original data")
    parser.add_argument("--residual_mode", type=bool, default=True, choices=[True, False],
                        help="Whether to use indirect prediction mode")
    parser.add_argument("--scale", type=float, default=1000.0,
                        help="Whether to use indirect prediction mode")
    input_args = parser.parse_args()
    return input_args


def inference():
    """Inference process using trained models"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    summary_params = config["summary"]

    # prepare dataset
    infer_loader = create_dataset(data_params, is_infer=True, norm=args.norm, residual=args.residual_mode,
                                  scale=args.scale)
    dataset = infer_loader.batch(1, drop_remainder=False)

    # prepare model
    model = ResUnet3D(in_channels=model_params['in_dims'], base_channels=model_params['base'],
                      out_channels=model_params['out_dims'])

    summary_dir = os.path.join(summary_params['summary_dir'], f"norm-{args.norm}",
                               f"resi-{args.residual_mode} scale-{args.scale} {model_params['loss_fn']}")
    epoch = summary_params['epoch_load']
    param_dict = load_checkpoint(os.path.join(
        summary_dir, 'ckpt', f'ckpt-{epoch}.ckpt'))
    load_param_into_net(model, param_dict)
    model.set_train(False)
    problem = UnsteadyFlow3D(model, metric_fn=model_params['metric_fn'],
                             t_in=data_params['t_in'], t_out=data_params['t_out'],
                             residual=args.residual_mode, scale=args.scale)

    print("================================Start Inference================================", flush=True)
    infer_data_size = dataset.get_dataset_size()
    print(f'total infer steps: {infer_data_size}')
    it = dataset.create_tuple_iterator()
    inputs, labels = next(it)
    temp_inputs = inputs

    pred_list = []
    true_list = []
    for t in range(infer_data_size - 1):
        preds, updated_inputs = problem.step(temp_inputs)

        flow_preds = args.residual_mode * \
            temp_inputs[:, 0:data_params['t_out'], ...] + preds / args.scale
        pred_list.append(flow_preds.asnumpy().squeeze())
        flow_trues = args.residual_mode * \
            inputs[:, 0:data_params['t_out'], ...] + labels / args.scale
        true_list.append(flow_trues.asnumpy().squeeze())
        metric_step = RRMSE()(flow_preds, flow_trues)
        print(
            f"step: {t + 1:-2d}  cur_metric: {metric_step.asnumpy():.8f}", flush=True)

        temp_inputs = updated_inputs
        for _ in range(data_params['skip']):
            inputs, labels = next(it, -1)

    # Save results of the inference process
    infer_dir = os.path.join(summary_dir, "infer_results/")
    check_file_path(infer_dir)
    np.savez(infer_dir + './pred.npz', true=true_list, pred=pred_list)
    print("=================================End Inference=================================", flush=True)
    plot_results(save_path=infer_dir)


if __name__ == '__main__':
    print("pid:", os.getpid())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id)
    inference()
