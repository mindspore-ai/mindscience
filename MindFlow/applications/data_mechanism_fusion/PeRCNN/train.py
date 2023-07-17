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
"""train"""
import argparse
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt

from mindspore import ops, context, nn, set_seed, save_checkpoint, jit, load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
from mindflow.utils import load_yaml_config

from src import RecurrentCNNCell, UpScaler, RecurrentCNNCellBurgers
from src import Trainer
from src import post_process_v2

set_seed(123456)
np.random.seed(123456)


def train(trainer, stage, config):
    """train"""
    if 'milestone_num' in config.keys():
        milestone = list([(config['epochs']//config['milestone_num'])*(i + 1)
                          for i in range(config['milestone_num'])])
        learning_rate = config['learning_rate']
        lr = float(config['learning_rate'])*np.array(list([config['gamma']
                                                           ** i for i in range(config['milestone_num'])]))
        learning_rate = nn.piecewise_constant_lr(milestone, list(lr))
    else:
        learning_rate = config['learning_rate']

    if stage == 'pretrain':
        params = trainer.upconv.trainable_params()
    else:
        params = trainer.upconv.trainable_params(
        ) + trainer.recurrent_cnn.trainable_params()

    optimizer = nn.Adam(params, learning_rate=learning_rate)

    def forward_fn():
        if stage == 'pretrain':
            return trainer.get_ic_loss()
        return trainer.get_loss()

    if stage == 'pretrain':
        grad_fn = ops.value_and_grad(forward_fn, None, params, has_aux=False)
    else:
        grad_fn = ops.value_and_grad(forward_fn, None, params, has_aux=True)

    @ jit
    def train_step():
        res, grads = grad_fn()
        res = ops.depend(res, optimizer(grads))
        return res

    best_loss = 100000
    for epoch in range(1, 1 + config['epochs']):
        time_beg = time.time()
        upconv.set_train(True)
        recurrent_cnn.set_train(True)
        if stage == 'pretrain':
            step_train_loss = train_step()
            print(
                f"epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg) :.3f} s")
        else:
            step_train_loss, loss_data, loss_ic, loss_phy, loss_valid = train_step()
            print(f"epoch: {epoch} train loss: {step_train_loss} ic_loss: {loss_ic} data_loss: {loss_data}\
                   val_loss: {loss_valid} phy_loss: {loss_phy} epoch time: {(time.time() - time_beg): .3f} s")
            if step_train_loss < best_loss:
                best_loss = step_train_loss
                print('best loss', best_loss, 'save model')
                save_checkpoint(upconv, "./model/" + args.pattern +
                                '/' + config['name_conf'] + "_upconv.ckpt")
                save_checkpoint(recurrent_cnn, "./model/" + args.pattern +
                                '/' + config['name_conf'] + "_recurrent_cnn.ckpt")
    if args.pattern == 'physics_driven':
        trainer.recurrent_cnn.show_coef()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="burgers train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=2,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./percnn_burgers.yaml")
    parser.add_argument("--pattern", type=str, default="data_driven")
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id,
                        max_call_depth=99999999)
    print(
        f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    burgers_config = load_yaml_config(args.config_file_path)[args.pattern]

    upconv = UpScaler(in_channels=burgers_config['input_channel'],
                      out_channels=burgers_config['out_channels'],
                      hidden_channels=burgers_config['upscaler_hidden_channel'],
                      kernel_size=burgers_config['kernel_size'],
                      stride=burgers_config['stride'],
                      has_bais=True)

    if args.pattern == 'data_driven':
        recurrent_cnn = RecurrentCNNCell(input_channels=burgers_config['input_channel'],
                                         hidden_channels=burgers_config['rcnn_hidden_channel'],
                                         kernel_size=burgers_config['kernel_size'],
                                         compute_dtype=mstype.float32)
    else:
        recurrent_cnn = RecurrentCNNCellBurgers(kernel_size=burgers_config['kernel_size'],
                                                init_coef=burgers_config['init_coef'],
                                                compute_dtype=mstype.float32)

    percnn_trainer = Trainer(upconv=upconv,
                             recurrent_cnn=recurrent_cnn,
                             timesteps_for_train=burgers_config['use_timestep'],
                             dx=burgers_config['dx'],
                             dt=burgers_config['dy'],
                             nu=burgers_config['nu'],
                             compute_dtype=mstype.float32)

    param_dict = load_checkpoint('model/data_driven/train_upconv.ckpt')
    param_not_load, _ = load_param_into_net(upconv, param_dict)
    param_dict = load_checkpoint('model/data_driven/train_recurrent_cnn.ckpt')
    param_not_load, _ = load_param_into_net(recurrent_cnn, param_dict)

    output = percnn_trainer.get_output(1800)
    output = ops.concat((output, output[:, :, :, 0:1]), axis=3)
    output = ops.concat((output, output[:, :, 0:1, :]), axis=2)
    truth_clean = np.concatenate(
        (percnn_trainer.truth_clean, percnn_trainer.truth_clean[:, :, :, 0:1]), axis=3)
    truth_clean = np.concatenate(
        (truth_clean, truth_clean[:, :, 0:1, :]), axis=2)
    low_res = truth_clean[:, :, ::2, ::2]
    output = output.asnumpy()

    print(output.shape, truth_clean.shape, low_res.shape)

    err_list = []
    img_path = []
    for i in range(0, 1801, 10):
        err, fig_path = post_process_v2(output, truth_clean, low_res, xmin=0, xmax=1, ymin=0, ymax=1,
                                        num=i, fig_save_path='figures_' + args.pattern)
        print('infer step:', i, ', relative l2 error', err)
        err_list.append([i, err])
        img_path.append(fig_path)
    gif_images = []
    for path in img_path:
        gif_images.append(imageio.imread(path))
    imageio.mimsave('results.gif', gif_images, duration=0.01)
    err_list = np.array(err_list)

    plt.figure(figsize=(6, 4))
    plt.plot(err_list[:, 0], err_list[:, 1])
    plt.xlabel('infer_step')
    plt.ylabel('relative_l2_error')
    plt.savefig("error.png")
