# Copyright 2021 Huawei Technologies Co., Ltd
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
import timeit

import yaml
import numpy as np
import mindspore as ms
from mindspore import nn

from src.customloss import CustomWithLossCell, CustomWithEvalCell, CustomWithEval2Cell, alpha_true_func
from src.data_gen import s_x, s_z, t01, t02, t_la, N1, N2, N3, N4
from src.data_gen import train_dataset, dataset01, dataset02, dataset2, dataset_seism
from src.data_gen import xx, zz, az, d_s, X_S, u_ini1x, u_ini1z, u_ini2x, u_ini2z, u_specx, u_specz
from src.model import Net, Net0
from utils.plot import plot_ini_total_disp_spec_sumevents, plot_sec_wavefield_input_spec_sumevents
from utils.plot import plot_total_disp_spec_testdata_sumevents
from utils.plot import plot_total_predicted_dispfield_and_diff, plot_inverted_alpha, plot_misfit, plot_seismogram
from utils.plot import plot_true_wavespeed, plot_ini_guess_wavespeed


def parse_args():
    """parse args"""
    with open('src/default_config.yaml', 'r') as y:
        cfg = yaml.full_load(y)
    return cfg


def train(args):
    """train"""
    plot_ini_total_disp_spec_sumevents(xx, zz, u_ini1x, u_ini1z, t01)
    plot_sec_wavefield_input_spec_sumevents(xx, zz, u_ini2x, u_ini2z, t02)
    plot_total_disp_spec_testdata_sumevents(xx, zz, u_specx, u_specz, t_la, t01)

    layers = [3] + [100] * 8 + [1]
    neural_net = Net(layers=layers)
    layers0 = [2] + [20] * 5 + [1]
    neural_net0 = Net0(layers=layers0)

    net_with_loss = CustomWithLossCell(
        args, neural_net, neural_net0, u_ini1x, u_ini1z, u_ini2x, u_ini2z, s_x, s_z, N1, N2, N3, N4)
    group_params = [{'params': neural_net.trainable_params()},
                    {'params': neural_net0.trainable_params()}]
    optim = nn.Adam(group_params, learning_rate=args["learning_rate"], eps=args["eps"])

    train_net = nn.TrainOneStepCell(net_with_loss, optim)

    loss_eval = np.zeros((1, 7))
    loss_rec = np.empty((0, 7))

    alpha_true0 = alpha_true_func(dataset01, args)
    alpha_true0 = alpha_true0.reshape(xx.shape)

    eval_net = CustomWithEvalCell(args, neural_net, neural_net0, u_ini1x, u_ini1z, u_ini2x, u_ini2z, s_x, s_z, N1, N2,
                                  N3, N4)
    eval_net.set_train(False)

    eval_net2 = CustomWithEval2Cell(neural_net=neural_net, neural_net0=neural_net0)
    eval_net2.set_train(False)
    _, _, alpha_plot = eval_net2(dataset01)
    plot_true_wavespeed(xx, zz, alpha_true0, X_S)
    alpha_plot = alpha_plot.reshape(xx.shape)
    plot_ini_guess_wavespeed(xx, zz, alpha_plot)

    start = timeit.default_timer()
    epoch = -1

    for d in train_dataset.create_dict_iterator():
        train_data = ms.Tensor(d["data"], dtype=ms.float32)
        for _ in range(200):
            epoch = epoch + 1
            _ = train_net(train_data)

        stop = timeit.default_timer()
        print('Time: ', stop - start)

        eval_data = ms.Tensor(d["data"], dtype=ms.float32)

        loss_collection = eval_net(eval_data)

        loss_val = loss_collection[0]
        loss_pde_val = loss_collection[1]
        loss_init_disp1_val = loss_collection[2]
        loss_init_disp2_val = loss_collection[3]
        loss_seism_val = loss_collection[4]
        loss_bc_val = loss_collection[5]

        print('Epoch: ', epoch, ', Loss: ', loss_val, ', Loss_pde: ',
              loss_pde_val, ', Loss_init_disp1: ', loss_init_disp1_val)
        print(', Loss_init_disp2: ', loss_init_disp2_val,
              'Loss_seism: ', loss_seism_val, 'Loss_stress: ', loss_bc_val)

        ux01, uz01, alpha0 = eval_net2(dataset01)
        ux02, uz02, _ = eval_net2(dataset02)
        uxt, uzt, _ = eval_net2(dataset2)
        uz_seism_pred, ux_seism_pred, _ = eval_net2(dataset_seism)

        loss_eval[0, 0] = epoch
        loss_eval[0, 1] = loss_val
        loss_eval[0, 2] = loss_pde_val
        loss_eval[0, 3] = loss_init_disp1_val
        loss_eval[0, 4] = loss_init_disp2_val
        loss_eval[0, 5] = loss_seism_val
        loss_eval[0, 6] = loss_bc_val

        loss_rec = np.concatenate((loss_rec, loss_eval), axis=0)

        plot_total_predicted_dispfield_and_diff(
            xx, zz, ux01, uz01, ux02, uz02, uxt, uzt, u_specx, u_specz, t01, t02, t_la)

        plot_inverted_alpha(xx, zz, alpha0, alpha_true0)

        plot_misfit(loss_rec)

        plot_seismogram(X_S, s_z, s_x, uz_seism_pred, ux_seism_pred, az, d_s)

        ms.save_checkpoint(neural_net, "MyNet.ckpt")
        ms.save_checkpoint(neural_net0, "MyNet0.ckpt")


if __name__ == "__main__":
    cfg_ = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    train(cfg_)
