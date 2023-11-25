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
"""eval of CaeNet"""
import os
import argparse

import numpy as np
import mindspore.common.dtype as mstype

from mindspore import load_checkpoint, load_param_into_net, set_seed, Tensor, ops, context
from mindflow.utils import load_yaml_config
from src import create_cae_dataset, CaeNet1D, CaeNet2D, plot_cae_eval

np.random.seed(0)
set_seed(0)


def cae_eval(config_file_path, case):
    """eval of CaeNet"""
    # prepare params
    config = load_yaml_config(config_file_path)
    if case in ('sod', 'shu_osher'):
        data_params = config["1D_cae_data"]
        model_params = config["1D_cae_model"]
        prediction_params = config["1D_prediction"]
    else:
        data_params = config["2D_cae_data"]
        model_params = config["2D_cae_model"]
        prediction_params = config["2D_prediction"]

    # prepare network
    if case in ('sod', 'shu_osher'):
        cae = CaeNet1D(model_params["data_dimension"], model_params["conv_kernel_size"],
                       model_params["maxpool_kernel_size"], model_params["maxpool_stride"],
                       model_params["encoder_channels"], model_params["decoder_channels"])
    else:
        cae = CaeNet2D(model_params["data_dimension"], model_params["conv_kernel_size"],
                       model_params["maxpool_kernel_size"], model_params["maxpool_stride"],
                       model_params["encoder_channels"], model_params["decoder_channels"],
                       model_params["channels_dense"])

    cae_param_dict = load_checkpoint(prediction_params["cae_ckpt_path"])
    load_param_into_net(cae, cae_param_dict)

    # prepare dataset
    _, true_data = create_cae_dataset(data_params["data_path"], data_params["batch_size"], data_params["multiple"])
    true_data_multiple = true_data * data_params["multiple"]
    data_set = np.expand_dims(true_data_multiple, 1).astype(np.float32)

    print(f"=================Start CaeNet eval=====================")
    if case in ('sod', 'shu_osher'):
        encoded = cae.encoder(Tensor(data_set))
        cae_predict = np.squeeze(cae(Tensor(data_set)).asnumpy())
    else:
        encoded = ops.zeros((data_params["time_size"], model_params["latent_size"]), mstype.float32)
        cae_predict = np.zeros(true_data.shape)
        for i in range(prediction_params["encoder_data_split"]):
            time_predict_start, time_predict_end = \
                prediction_params["encoder_time_spilt"][i], prediction_params["encoder_time_spilt"][i + 1]
            encoded[time_predict_start: time_predict_end] = \
                cae.encoder(Tensor(data_set[time_predict_start: time_predict_end]))
            cae_predict[time_predict_start: time_predict_end] = \
                np.squeeze(cae(Tensor(data_set[time_predict_start: time_predict_end])).asnumpy())
    print(f"===================End CaeNet eval====================")

    cae_error_mean = plot_cae_eval(encoded, cae_predict, true_data, data_params["multiple"],
                                   prediction_params["prediction_result_dir"], data_params["time_size"])
    print("Cae prediction mean error: " + str(cae_error_mean))
    return encoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CaeNet eval')
    parser.add_argument("--case", type=str, default="sod",
                        choices=["sod", "shu_osher", "riemann", "kh", "cylinder"],
                        help="Which case to run, support 'sod', 'shu_osher', 'riemann', 'kh', 'cylinder")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "CPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    print(f"pid:{os.getpid()}")
    cae_eval(args.config_file_path, args.case)
