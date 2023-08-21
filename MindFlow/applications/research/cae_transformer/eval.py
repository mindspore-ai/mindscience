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
"""prediction process"""
import os
import argparse
import yaml
import numpy as np
from matplotlib import pyplot as plt

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net, set_seed, nn
from src import create_caetransformer_dataset, CaeInformer


np.random.seed(0)
set_seed(0)


def cae_transformer_prediction(input_args):
    """Process of prediction with cae-transformer net"""
    # prepare params
    with open(input_args.config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    data_params = config["data"]
    model_params = config["cae_transformer"]
    prediction_params = config["prediction"]

    # prepare network
    model = CaeInformer(**model_params)
    model_param_dict = load_checkpoint(prediction_params["model_ckpt_path"])
    load_param_into_net(model, model_param_dict)

    # prepare dataset
    _, eval_data = create_caetransformer_dataset(
        data_params['data_path'],
        data_params["batch_size"],
        data_params["seq_len"],
        data_params["pred_len"],
    )
    l = data_params['input_dim']
    eval_data = np.reshape(eval_data, (-1, 1, l, l))
    plot_dir = prediction_params['prediction_result_dir']
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    print(f"=================Start cae-transformer prediction=====================")
    model.set_train(False)
    input_seq = eval_data[:data_params['seq_len']]
    input_seq = np.reshape(input_seq, (1, -1, 1, l, l))
    output = model(ms.Tensor(input_seq, ms.float32))
    output = output.asnumpy()
    output = np.reshape(output, (l, l))
    plt.subplot(1, 2, 1)
    plt.contourf(output)
    plt.title('output')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(1, 2, 2)
    gt = np.reshape(eval_data[data_params['seq_len']], (256, 256))
    plt.contourf(gt)
    plt.title('ground truth')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/prediction_result.jpg')
    plt.show()
    model.set_train(True)
    print(f"===================End transformer prediction====================")


def cae_transformer_eval(model, eval_data, data_params):
    """Process of prediction with cae-transformer net"""

    l = data_params['input_dim']
    eval_data = np.reshape(eval_data, (-1, 1, l, l))
    loss_fn = nn.MSELoss()
    print(f"=================Start cae-transformer evaluation=====================")
    eval_data_size = eval_data.shape[0] - data_params['seq_len'] - data_params['pred_len']
    eval_losses = np.zeros((eval_data_size))
    for i in range(eval_data_size):
        input_seq = eval_data[i:i+data_params['seq_len']]
        input_seq = np.reshape(input_seq, (1, -1, 1, l, l))
        output = model(ms.Tensor(input_seq, ms.float32))
        output = output.asnumpy()
        output = np.reshape(output, (l, l))
        eval_loss = loss_fn(ms.Tensor(output, ms.float32),
                            ms.Tensor(eval_data[i+data_params['seq_len'], 0], ms.float32))
        eval_losses[i] = eval_loss
    print(f"eval loss: {np.mean(eval_losses)}")
    print(f"===================End transformer evaluation====================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cae-transformer prediction")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    print(f"pid:{os.getpid()}")
    cae_transformer_prediction(args)
