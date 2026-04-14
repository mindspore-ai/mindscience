
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

"""plotting results"""
import os
from math import sqrt

import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import ops, Tensor
import numpy as np

from sciai.architecture import MLP, Swish
from sciai.utils import print_log


def plot_contour(args, *inputs):
    """Plot contour"""
    u_all, u_cfd, x, y, case_idx, scale = inputs
    # aneurysm
    plot_x = 0.8
    plot_y = 0.06
    fontsize = 14
    axis_limit = [0, 1, -0.15, 0.15]

    plt.figure()
    plt.subplot(212)
    plt.scatter(x, y, c=u_all[:, 0], vmin=min(u_cfd[:, 0]), vmax=max(u_cfd[:, 0]))
    plt.text(plot_x, plot_y, r'DNN', {'color': 'b', 'fontsize': fontsize})
    plt.axis(axis_limit)
    plt.colorbar()
    plt.subplot(211)
    plt.scatter(x, y, c=u_cfd[:, 0], vmin=min(u_cfd[:, 0]), vmax=max(u_cfd[:, 0]))
    plt.colorbar()
    plt.text(plot_x, plot_y, r'CFD', {'color': 'b', 'fontsize': fontsize})
    plt.axis(axis_limit)
    plt.savefig(f"{args.figures_path}/{int(case_idx)}scale{scale}uContour_test.png", bbox_inches='tight')
    print_log(f"path is: {args.figures_path}/{int(case_idx)}scale{scale}uContour_test.png")

    plt.figure()
    plt.subplot(212)
    plt.scatter(x, y, c=u_all[:, 1], vmin=min(u_cfd[:, 1]), vmax=max(u_cfd[:, 1]))
    plt.text(plot_x, plot_y, r'DNN', {'color': 'b', 'fontsize': fontsize})
    plt.axis(axis_limit)
    plt.colorbar()
    plt.subplot(211)
    plt.scatter(x, y, c=u_cfd[:, 1], vmin=min(u_cfd[:, 1]), vmax=max(u_cfd[:, 1]))
    plt.colorbar()
    plt.text(plot_x, plot_y, r'CFD', {'color': 'b', 'fontsize': fontsize})
    plt.axis(axis_limit)
    plt.savefig(f"{args.figures_path}/{int(case_idx)}scale{scale}vContour_test.png", bbox_inches='tight')
    plt.close('all')
    plt.show()


def plot_wall_shear(args, case_idx, w_ctl, w_ctl_ml):
    """Plot wall shear"""
    data_cfd_wss = load_with_default(args, f"{case_idx}CFD_wss.npz")
    unique_x = data_cfd_wss['x']
    wall_shear_mag_up = data_cfd_wss['wss']
    data_nn_wss = load_with_default(args, f"{case_idx}NN_wss.npz")
    n_nwall_shear_mag_up = data_nn_wss['wss']
    # show plot
    plt.figure()
    plt.plot(unique_x, wall_shear_mag_up, label='CFD', color='darkblue', linestyle='-', lw=3.0, alpha=1.0)
    plt.plot(unique_x, n_nwall_shear_mag_up, label='DNN', color='red', linestyle='--', dashes=(5, 5), lw=2.0,
             alpha=1.0)
    plt.xlabel(r'x', fontsize=16)
    plt.ylabel(r'$\tau_{c}$', fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig(f"{args.figures_path}/{int(case_idx)}nu{args.nu}wallShear_test.png", bbox_inches='tight')
    plt.close('all')
    # show center wall shear
    # CFD
    w_ctl[int(case_idx - 1)] = wall_shear_mag_up[int(len(wall_shear_mag_up) / 2)]
    # NN
    w_ctl_ml[int(case_idx - 1)] = n_nwall_shear_mag_up[int(len(n_nwall_shear_mag_up) / 2)]


def det_test(x, y, scale, case_idx, args):
    """Det test"""
    d_p = 0.1
    mu = 0.5
    sigma = 0.1
    # Geometry cases
    swish = Swish()
    net_u = MLP(args.layers, activation=swish)
    net_v = MLP(args.layers, activation=swish)
    net_p = MLP(args.layers, activation=swish)
    ms.load_checkpoint(args.load_ckpt_path[0], net_u)
    ms.load_checkpoint(args.load_ckpt_path[1], net_v)
    ms.load_checkpoint(args.load_ckpt_path[2], net_p)
    print_log("ckpts are loaded successfully")
    print_log(f"ckpt path: {args.load_ckpt_path}")
    scale = Tensor([scale], dtype=ms.float32)
    const = Tensor([1 / sqrt(2 * np.pi * sigma ** 2)], dtype=ms.float32)
    r = scale * const * ops.exp(Tensor(-(x - mu) ** 2 / (2 * sigma ** 2), dtype=ms.float32))
    r_inlet = 0.05
    y_up = r_inlet - r

    xt = Tensor(x, dtype=ms.float32)
    yt = Tensor(y, dtype=ms.float32)
    print_log("tensors are set successfully")
    xt = xt.view(len(xt), -1)
    yt = yt.view(len(yt), -1)
    print_log("tensors are viewed successfully")
    ones_like = ops.OnesLike()
    scalet = scale * Tensor(ones_like(xt), dtype=ms.float32)
    rt = Tensor(y_up).astype(ms.float32)
    rt = rt.view(len(rt), -1)
    net_in = ops.cat((xt, yt, scalet), axis=1)

    u_t = net_u(net_in)
    v_t = net_v(net_in)
    p_t = net_p(net_in)

    u_hard = u_t * (rt ** 2 - yt ** 2)
    v_hard = (rt ** 2 - yt ** 2) * v_t
    l = 1
    x_start = 0
    x_end = l
    p_hard = (x_start - xt) * 0 + d_p * (x_end - xt) / l + 0 * yt + (x_start - xt) * (x_end - xt) * p_t

    u_hard = u_hard.asnumpy()
    v_hard = v_hard.asnumpy()
    p_hard = p_hard.asnumpy()
    np.savez(args.save_data_path + str(int(case_idx)) + 'ML_WallStress_uvp', x_center=x, y_center=y, u_center=u_hard,
             v_center=v_hard, p_center=p_hard)

    return u_hard, v_hard, p_hard


def load_with_default(args, filename):
    """load data in args.save_data_path. If not found, it would load default one in args.load_data_path"""
    data_path = f"{args.save_data_path}/{filename}"
    if not os.path.exists(data_path):
        data_path = f"{args.load_data_path}/{filename}"
        print_log(f"No generated '{filename}' found, trying to load pre-trained result by default.")
    data = np.load(data_path)
    return data
