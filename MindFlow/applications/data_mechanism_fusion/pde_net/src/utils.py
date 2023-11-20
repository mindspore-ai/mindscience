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
"""
logger and check file function
"""
import os
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import mindspore
from mindspore import ops, Tensor
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindflow.loss import RelativeRMSELoss
from mindflow.pde import UnsteadyFlowWithLoss
from mindflow.cell import PDENet
from mindflow.utils import print_log


def init_model(config):
    return PDENet(height=config["mesh_size"],
                  width=config["mesh_size"],
                  channels=config["channels"],
                  kernel_size=config["kernel_size"],
                  max_order=config["max_order"],
                  dx=2 * np.pi / config["mesh_size"],
                  dy=2 * np.pi / config["mesh_size"],
                  dt=config["dt"],
                  periodic=config["perodic_padding"],
                  enable_moment=config["enable_moment"],
                  if_fronzen=config["if_frozen"],
                  )


def calculate_lp_loss_error(problem, test_dataset, batch_size):
    """Calculates loss error"""
    print_log(
        "================================Start Evaluation================================")
    time_beg = time.time()
    lploss_error = 0.0
    max_error = 0.0
    length = test_dataset.get_dataset_size()
    for data in test_dataset.create_dict_iterator():
        test_label = data["uT"]
        test_data = data["u0"]
        lploss_error_step = problem.get_loss(
            test_data, test_label) / batch_size
        lploss_error += lploss_error_step

        if lploss_error_step >= max_error:
            max_error = lploss_error_step

    lploss_error = lploss_error / length
    print_log(f"LpLoss_error: {lploss_error}")
    print_log(
        "=================================End Evaluation=================================")
    print_log(f"predict total time: {time.time() - time_beg}s")


def scheduler(lr_scheduler_step, step, lr):
    if step % lr_scheduler_step == 0:
        lr *= 0.5
        print_log(f"learning rate reduced to {lr}")
    return lr


def make_dir(path):
    """ make directory"""
    if os.path.exists(path):
        return

    try:
        permissions = os.R_OK | os.W_OK | os.X_OK
        os.umask(permissions << 3 | permissions)
        mode = permissions << 6
        os.makedirs(path, mode=mode, exist_ok=True)
    except PermissionError as e:
        mindspore.log.critical(
            "No write permission on the directory(%r), error = %r", path, e)
        raise TypeError("No write permission on the directory.") from e
    finally:
        pass


def get_param_dic(summary_dir, current_step, epochs):
    file_dir = "step_{}/pdenet-{}.ckpt".format(current_step, epochs)
    checkpoint_dir = os.path.join(summary_dir, file_dir)
    param_dict = load_checkpoint(checkpoint_dir)
    return param_dict


def plot_coe(coes, img_dir, prefix="coe", step=0, title="coes"):
    """plot coefficients of PDE"""
    make_dir(img_dir)
    plt.rcParams['figure.figsize'] = (12, 3)
    # if all coffient need plot, then num_coe, _, _ = np.shape(coes)
    num_coe = 5
    coes_2d = [coes[idx, :, :] for idx in range(num_coe)]
    fig = plt.figure()
    plt.title(title)
    plt.axis('off')
    gs = gridspec.GridSpec(1, num_coe)
    gs_idx = 0

    for idx, coe in enumerate(coes_2d):
        ax = fig.add_subplot(gs[gs_idx])
        gs_idx += 1
        try:
            coe = coe.asnumpy()
        except AttributeError:
            pass

        if idx < 2:
            img = ax.imshow(coe.T, vmin=coe.min(), vmax=coe.max(),
                            cmap=plt.get_cmap("turbo"), origin='lower')
        else:
            img = ax.imshow(coe.T, vmin=-0.75, vmax=0.75,
                            cmap=plt.get_cmap("turbo"), origin='lower')
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


def get_label_coe(max_order, resolution):
    """labels of PDE coefficients"""
    x_cord = np.linspace(0, 2 * math.pi, num=resolution)
    y_cord = np.linspace(0, 2 * math.pi, num=resolution)
    mesh_x, mesh_y = np.meshgrid(x_cord, y_cord)

    coe_dict = dict()
    coe_dict['10'] = 0.5 * (np.cos(mesh_y) + mesh_x *
                            (2 * math.pi - mesh_x) * np.sin(mesh_x)) + 0.6
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
                coes.append(coe_dict.get("{:.0f}{:.0f}".format(ord_j, ord_i)))
            elif idx != 0:
                coes.append(others)
            idx += 1
    coes = np.stack(coes, axis=0).reshape(-1, resolution, resolution)
    return coes


def _get_mesh_grid(mesh_size):
    x = np.linspace(0, 2 * math.pi, num=mesh_size)
    y = np.linspace(0, 2 * math.pi, num=mesh_size)
    x_grid, y_grid = np.meshgrid(x, y)
    x_grid = x_grid.reshape((-1, 1))
    y_grid = y_grid.reshape((-1, 1))
    return x_grid, y_grid


def plot_test_error(problem, loss_fn, item, step, mesh_size, figure_out_dir="./figure"):
    """Plot the test error"""
    make_dir(os.path.join(figure_out_dir, "coes"))
    make_dir(os.path.join(figure_out_dir, "error_test"))

    x_grid, y_grid = _get_mesh_grid(mesh_size)
    u0 = item["u0"].asnumpy()
    u_t = item["u_step" + str(step)].asnumpy()
    x = Tensor(u0.reshape(1, 1, 1, mesh_size, mesh_size), dtype=mstype.float32)
    y = Tensor(u_t.reshape(1, 1, mesh_size, mesh_size), dtype=mstype.float32)
    y_predict = problem.step(x)[:, -1, ...]
    print_log("sample {}, MSE Loss {}".format(step, loss_fn(y_predict, y)))
    error = y_predict - y

    plt.figure(figsize=(16, 4))

    plt_y = y.asnumpy()[0, 0, :, :].reshape(-1, 1)
    plt.subplot(1, 3, 1)
    plt.title("label")
    plt.scatter(x_grid, y_grid, c=plt_y, cmap=plt.cm.rainbow,
                vmin=min(plt_y[:]), vmax=max(plt_y[:]))
    plt.colorbar()

    plt_y_predict = y_predict.asnumpy()[0, 0, :, :].reshape(-1, 1)
    plt.subplot(1, 3, 2)
    plt.title("prediction")
    plt.scatter(x_grid, y_grid, c=plt_y_predict, cmap=plt.cm.rainbow, vmin=min(plt_y_predict[:]),
                vmax=max(plt_y_predict[:]))
    plt.colorbar()

    error_val = error.asnumpy()[0, 0, :, :].reshape(-1, 1)
    plt.subplot(1, 3, 3)
    plt.title("error")
    plt.scatter(x_grid, y_grid, c=error_val, cmap=plt.cm.rainbow,
                vmin=min(error_val[:]), vmax=max(error_val[:]))
    plt.colorbar()

    plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
    plt.savefig(os.path.join(figure_out_dir, 'error_test/result.jpg'))


def _extapolation(config, extra_step, test_data_iterator):
    """long time prediction test for given ckpt"""
    model_params = config["model"]
    summary_params = config["summary"]
    optimizer_params = config["optimizer"]

    model = init_model(model_params)
    if extra_step == 1:
        model.if_fronzen = True
    param_dict = get_param_dic(
        os.path.join(summary_params["root_dir"],
                     summary_params["ckpt_dir"]),
        optimizer_params["multi_step"],
        optimizer_params["epochs"])
    load_param_into_net(model, param_dict)

    cast = ops.Cast()
    problem = UnsteadyFlowWithLoss(
        model, t_out=extra_step, loss_fn=RelativeRMSELoss(), data_format="NTCHW")

    error_list = []
    for item in test_data_iterator:
        u0 = item["u0"]
        ut = item["u_step{:.0f}".format(extra_step)]
        u0 = cast(u0, mstype.float32)
        ut = cast(ut, mstype.float32)
        u0 = u0.reshape(-1, 1, 1,
                        model_params["mesh_size"], model_params["mesh_size"])
        ut = ut.reshape(-1, 1, 1,
                        model_params["mesh_size"], model_params["mesh_size"])
        error_list.append(problem.get_loss(u0, ut).asnumpy().reshape(1)[0])
    return error_list


def plot_extrapolation_error(config, dataset, max_step=40):
    """ plot extrapolation error """
    error_data = []
    plot_data = np.zeros([max_step, 3])
    for i in range(1, max_step + 1):
        test_dataset = dataset.create_test_dataset(i)
        test_data_iterator = test_dataset.create_dict_iterator()
        error = _extapolation(config, i, test_data_iterator)
        error_data.append(error)
        p25 = np.percentile(error, 25)
        p75 = np.percentile(error, 75)
        print_log(
            "step = {:.0f}, p25 = {:.5f}, p75 = {:.5f}".format(i, p25, p75))
        plot_data[i - 1, :] = [i, p25, p75]

    summary_params = config["summary"]
    plt.semilogy(plot_data[:, 0], plot_data[:, 1], color='orange')
    plt.semilogy(plot_data[:, 0], plot_data[:, 2], color='orange')
    plt.fill_between(plot_data[:, 0], plot_data[:, 1],
                     plot_data[:, 2], facecolor='orange', alpha=0.5)
    plt.xlim(1, max_step)
    plt.ylim(0.01, 100)
    plt.savefig(os.path.join(
        summary_params["root_dir"], summary_params["visualization_dir"], 'extrapolation.jpg'))
