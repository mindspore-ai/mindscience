# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
plot the contour of pde net predictions and label, plot the pde net coe and label
"""
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from mindspore import Tensor, context, nn, ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
from mindflow.cell.neural_operators import PDENet

from src.dataset import DataPrepare
from src.data_generator import DataGenerator
from src.utils import check_file_path
from config import train_config as config

parser = argparse.ArgumentParser(description="visualization")
parser.add_argument('--ckpt_step', type=int, default=20, help="step of ckpt")
parser.add_argument('--ckpt_epoch', type=int, default=500, help="epoch of ckpt")
parser.add_argument('--step', type=int, default=20, help="test step")
parser.add_argument('--test_data_size', type=int, default=16, help="data size for test")
parser.add_argument('--device_target', type=str, default="Ascend", help="device target")
parser.add_argument('--device_id', type=int, default=0, help="device id")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target=args.device_target,
                    device_id=args.device_id
                    )


def plot_coe(coes, img_dir, prefix="coe", step=0):
    """plot coefficients of PDE"""
    plt.rcParams['figure.figsize'] = (12, 9)
    num_coe, _, _ = np.shape(coes)
    coes_2d = [coes[idx, :, :] for idx in range(num_coe)]
    fig = plt.figure()
    gs = gridspec.GridSpec(3, math.ceil(num_coe / 3))
    gs_idx = 0

    for idx, coe in enumerate(coes_2d):
        ax = fig.add_subplot(gs[gs_idx])
        gs_idx += 1
        try:
            coe = coe.asnumpy()
        except AttributeError:
            pass

        if idx < 2:
            img = ax.imshow(coe.T, vmin=coe.min(), vmax=coe.max(), cmap=plt.get_cmap("turbo"), origin='lower')
        else:
            img = ax.imshow(coe.T, vmin=-0.75, vmax=0.75, cmap=plt.get_cmap("turbo"), origin='lower')
        plt.axis('off')

        aspect = 20
        pad_fraction = 0.5
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cb = plt.colorbar(img, cax=cax)
        cb.ax.tick_params(labelsize=6)

    gs.tight_layout(fig, pad=0.2, w_pad=0.2, h_pad=0.2)
    if step > 0:
        file_name = os.path.join(img_dir, f"{prefix}_step-{step}.png")
    else:
        file_name = os.path.join(img_dir, f"{prefix}.png")
    fig.savefig(file_name)
    plt.close()


def coe_label(max_order, resolution):
    """labels of PDE coefficients"""
    x_cord = np.linspace(0, 2 * math.pi, num=resolution)
    y_cord = np.linspace(0, 2 * math.pi, num=resolution)
    mesh_x, mesh_y = np.meshgrid(x_cord, y_cord)

    coe_dict = dict()
    coe_dict['10'] = 0.5 * (np.cos(mesh_y) + mesh_x * (2 * math.pi - mesh_x) * np.sin(mesh_x)) + 0.6
    coe_dict['01'] = 2 * (np.cos(mesh_y) + np.sin(mesh_x)) + 0.8
    coe_dict['20'] = 0.2 * np.ones((resolution, resolution))
    coe_dict['02'] = 0.3 * np.ones((resolution, resolution))
    others = np.zeros((resolution, resolution))

    coes = []
    idx = 0
    for o1 in range(max_order + 1):
        for o2 in range(o1 + 1):
            ord_i = o1 - o2
            ord_j = o2
            if "{:.0f}{:.0f}".format(ord_i, ord_j) in coe_dict:
                coes.append(coe_dict.get("{:.0f}{:.0f}".format(ord_i, ord_j)))
            elif idx != 0:
                coes.append(others)
            idx += 1
    coes = np.stack(coes, axis=0).reshape(-1, resolution, resolution)
    return coes


if __name__ == '__main__':
    model = PDENet(height=config["mesh_size"],
                   width=config["mesh_size"],
                   channels=config["channels"],
                   kernel_size=config["kernel_size"],
                   max_order=config["max_order"],
                   step=args.step,
                   dx=2 * np.pi / config["mesh_size"],
                   dy=2 * np.pi / config["mesh_size"],
                   dt=config["dt"],
                   periodic=config["perodic_padding"],
                   enable_moment=config["enable_moment"],
                   if_fronzen=config["if_frozen"],
                   )
    if args.ckpt_step == 1:
        model.if_fronzen = True

    param_dict = load_checkpoint(
        "./summary_dir/summary/ckpt/step_{}/pdenet-{}_1.ckpt".format(args.ckpt_step, args.ckpt_epoch))
    load_param_into_net(model, param_dict)

    data = DataGenerator(step=args.step, config=config, mode="test", data_size=args.test_data_size,
                         file_name="data/eval.mindrecord")
    data.process()
    d = DataPrepare(config=config, data_file="data/eval.mindrecord")
    eval_dataset = d.test_data_prepare(args.step)
    iterator = eval_dataset.create_dict_iterator()
    cast = ops.Cast()

    x = np.linspace(0, 2 * math.pi, num=config["mesh_size"])
    y = np.linspace(0, 2 * math.pi, num=config["mesh_size"])
    x_grid, y_grid = np.meshgrid(x, y)
    x = x_grid.reshape((-1, 1))
    y = y_grid.reshape((-1, 1))

    check_file_path("figure")
    check_file_path("figure/coes")
    check_file_path("figure/error_test")
    loss_func = nn.MSELoss()
    cur_coe = model.coe.asnumpy()
    for i, item in enumerate(iterator):
        u0 = item["u0"].asnumpy()
        uT = item["u_step" + str(args.step)].asnumpy()
        x = Tensor(u0.reshape(1, 1, config["mesh_size"], config["mesh_size"]), dtype=mstype.float32)
        y = Tensor(uT.reshape(1, 1, config["mesh_size"], config["mesh_size"]), dtype=mstype.float32)
        y_predict = model(x)
        print("sample {}, MSE Loss {}".format(i, loss_func(y_predict, y)))
        error = y_predict - y

        plt.figure(figsize=(16, 4))

        plt_y = y.asnumpy()[0, 0, :, :].reshape(-1, 1)
        plt.subplot(1, 3, 1)
        plt.title("label")
        plt.scatter(x, y, c=plt_y, cmap=plt.cm.rainbow, vmin=min(plt_y[:]), vmax=max(plt_y[:]))
        plt.colorbar()

        plt_y_predict = y_predict.asnumpy()[0, 0, :, :].reshape(-1, 1)
        plt.subplot(1, 3, 2)
        plt.title("prediction")
        plt.scatter(x, y, c=plt_y_predict, cmap=plt.cm.rainbow, vmin=min(plt_y_predict[:]), vmax=max(plt_y_predict[:]))
        plt.colorbar()

        error_val = error.asnumpy()[0, 0, :, :].reshape(-1, 1)
        plt.subplot(1, 3, 3)
        plt.title("error")
        plt.scatter(x, y, c=error_val, cmap=plt.cm.rainbow, vmin=min(error_val[:]), vmax=max(error_val[:]))
        plt.colorbar()

        plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
        plt.savefig('figure/error_test/result{}.jpg'.format(i))

    coe_label = coe_label(max_order=config["max_order"], resolution=config["mesh_size"])
    coe_trained = model.coe
    plot_coe(coe_trained, './figure/coes', prefix="coe_trained", step=args.step)
    plot_coe(coe_label, './figure/coes', prefix="coe_label")
    print("coe plot completed!")
