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
"""train process"""
import os
import time
import argparse
import numpy as np

from mindspore import nn, ops, context, save_checkpoint, jit, data_sink, set_seed
import mindspore.common.dtype as mstype
from mindflow.utils import load_yaml_config
from src import create_dataset, EhdnnNet, HybridLoss, plot_train_loss

np.random.seed(0)
set_seed(0)

parser = argparse.ArgumentParser(description='eHDNN for Transonic buffet')
parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                    help="Context mode, support 'GRAPH', 'PYNATIVE'")
parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                    help="Whether to save intermediate compilation graphs")
parser.add_argument("--save_graphs_path", type=str, default="./summary")
parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                    help="The target device to run, support 'Ascend', 'GPU'")
parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
parser.add_argument("--train_aoa_list", type=int, nargs='+', default=[33],
                    help="The type for training, [33 ,34 , 35 , 36 , 37 , 38] for multi_state training /n"
                         "[33],....,[38] for single_state training")
parser.add_argument("--num_memory_layers", type=int, default=2, choices=[2, 4],
                    help="The number of layers of the whole Memory layer， 2 in single_state and 4 in multi_state")
parser.add_argument("--config_file_path", type=str, default="./config.yaml")

args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                    save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
                    device_target=args.device_target, device_id=args.device_id)
use_ascend = context.get_context(attr_key='device_target') == "Ascend"


def train():
    """train process"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]

    # prepare summary file
    summary_dir = optimizer_params["summary_dir"]
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    ckpt_dir = os.path.join(summary_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # prepare model
    model = EhdnnNet(model_params["in_channels"],
                     model_params["out_channels"],
                     model_params["num_layers"],
                     args.num_memory_layers,
                     model_params["kernel_size_conv"],
                     model_params["kernel_size_lstm"],
                     compute_dtype=mstype.float16 if use_ascend else mstype.float32
                     )
    loss_func = HybridLoss()
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=optimizer_params["lr"])

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')
    else:
        loss_scaler = None

    def forward_fn(x, y):
        pred = model(x)
        loss = loss_func(pred, y)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(x, y):
        loss, grads = grad_fn(x, y)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    @jit
    def eval_step(x, y):
        loss = forward_fn(x, y)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
        return loss

    # prepare dataset
    print(f"==================Load data sample ===================")
    dataset_train, dataset_eval = create_dataset(data_params["data_dir"],
                                                 data_params["data_length"],
                                                 data_params["train_ratio"],
                                                 args.train_aoa_list)
    print(f"======================End Load========================")

    # data sink
    train_sink_process = data_sink(train_step, dataset_train, sink_size=1)
    eval_sink_process = data_sink(eval_step, dataset_eval, sink_size=1)
    train_data_size, eval_data_size = dataset_train.get_dataset_size(), dataset_eval.get_dataset_size()

    print(f"====================Start train=======================")
    train_loss = []
    for epoch in range(1, optimizer_params["epochs"] + 1):
        local_time_beg = time.time()
        model.set_train(True)
        epoch_train_loss = 0
        for _ in range(train_data_size):
            epoch_train_loss = ops.squeeze(train_sink_process(), axis=())
        train_loss.append(epoch_train_loss)
        print(f"epoch: {epoch} train loss: {epoch_train_loss} epoch time: {time.time() - local_time_beg:.2f}s")
        if epoch % optimizer_params["eval_interval"] == 0:
            print(f"=================Start Evaluation=====================")
            model.set_train(False)
            eval_loss = []
            for _ in range(eval_data_size):
                step_eval_loss = ops.squeeze(eval_sink_process(), axis=())
                eval_loss.append(step_eval_loss)
            epoch_eval_loss = sum(eval_loss) / len(eval_loss)
            print(f"epoch: {epoch} eval loss: {epoch_eval_loss}")
            print(f"==================End Evaluation======================")

        if epoch % optimizer_params["save_ckpt_interval"] == 0:
            save_checkpoint(model, f"{ckpt_dir}/net_{epoch}.ckpt")
    print(f"=====================End train========================")
    plot_train_loss(train_loss, summary_dir, optimizer_params["epochs"])


if __name__ == "__main__":
    print(f"pid:{os.getpid()}")
    train()
