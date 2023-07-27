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
# ==============================================================================
"""
eval
"""
import copy
import os
import time

import numpy as np
import tqdm
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import Tensor, ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindelec.architecture import MultiScaleFCCell
from mindelec.common import PI
from mindelec.operators import SecondOrderGrad
from src.config import maxwell_3d_config


def load_config():
    """load config"""
    config = copy.deepcopy(maxwell_3d_config)
    return config


def evaluation(config):
    """evaluation"""
    tester = TestMaxwell3DSlab(config=config)
    tester.run()


def plot_waveguide(x, y, ez_pred, ez_true, ez_diff, ex_pred, ey_pred, save_dir=""):
    """
    Visualize the result of waveguide.

    Args:
        ez_pred: shape=(n,3), n is the number of sampling points. Each row is (x, y, E(x,y)),
            where (x, y) is accordinate and E(x, y) is the electric field intensity.
        ez_true: shape=(n,3)
        ez_diff: shape=(n,3)
        save_dir: the directory to save the result.
        ex_pred: shape=(n, 3)
        ey_pred: shape=(n, 3)
    """

    def adjust_xy_axis(ax):
        """Adjust the axis range and ticks"""
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_aspect(1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    ax0 = axes[0].scatter(x, y, c=ez_true, s=4, marker='s', cmap='jet')
    adjust_xy_axis(axes[0])
    axes[0].set_title("Ez ground truth")

    axes[1].scatter(x, y, c=ez_pred, s=4, marker='s', cmap='jet')
    adjust_xy_axis(axes[1])
    axes[1].set_title("Ez prediction")

    ax2 = axes[2].scatter(x, y, c=ez_diff, s=4, marker='s', cmap='binary')
    adjust_xy_axis(axes[2])
    axes[2].set_title("Ez difference")

    fig.colorbar(ax0, ax=[axes[0], axes[1]], shrink=0.8)
    fig.colorbar(ax2, ax=axes[2], shrink=0.8)

    plt.savefig(f"{save_dir}/waveguide_Ez.png", dpi=200, bbox_inches='tight')

    # Plot Ex, Ey
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    ax0 = axes[0].scatter(x, y, c=ex_pred, s=4, marker='s', cmap='binary')
    adjust_xy_axis(axes[0])
    axes[0].set_title("Ex difference")

    axes[1].scatter(x, y, c=ey_pred, s=4, marker='s', cmap='binary')
    adjust_xy_axis(axes[1])
    axes[1].set_title("Ey difference")

    fig.colorbar(ax0, ax=[axes[0], axes[1]], shrink=0.8)
    plt.savefig(f"{save_dir}/waveguide_Ex_Ey.png",
                dpi=200, bbox_inches='tight')


def plot_domain_plane(u1, u2, u3, xyrange, save_dir=""):
    """
    Visualize the domain plane.

    Args:
        u1: shape=(3,n,n), the result of first plane.
        u2: the result of second plane.
        u3: the result of third plane.
        xyrange: the range of coordinate.
        save_dir: the diectory to save images.
    """
    vmax = np.max([u1, u2, u3])
    vmin = np.min([u1, u2, u3])
    fig, axes = plt.subplots(4, 3, figsize=(10, 10))
    for e in range(3):
        ax = axes[0, e].imshow(u1[e], vmin=vmin, vmax=vmax, cmap='jet')
        axes[0, e].set_xticks([0, len(u1[e])], xyrange)
        axes[0, e].set_yticks([0, len(u1[e])], xyrange)
    for e in range(3):
        ax = axes[1, e].imshow(u2[e], vmin=vmin, vmax=vmax, cmap='jet')
        axes[1, e].set_xticks([0, len(u2[e])], xyrange)
        axes[1, e].set_yticks([0, len(u2[e])], xyrange)
    for e in range(3):
        ax = axes[2, e].imshow(u3[e], vmin=vmin, vmax=vmax, cmap='jet')
        axes[2, e].set_xticks([0, len(u3[e])], xyrange)
        axes[2, e].set_yticks([0, len(u3[e])], xyrange)
    fig.colorbar(ax, ax=[axes[e, xyz] for e in range(3) for xyz in range(3)], shrink=0.6)

    for e in range(3):
        axes[3, e].set_xticks([])
        axes[3, e].set_yticks([])
        axes[3, e].spines['top'].set_visible(False)
        axes[3, e].spines['right'].set_visible(False)
        axes[3, e].spines['bottom'].set_visible(False)
        axes[3, e].spines['left'].set_visible(False)
    text = plt.text(x=-2, y=0.5,
                    s='Top to down is 3 planes: x=0, y=0, z=0\nLeft to right are 3 components: $E_x$, $E_y$, $E_z$',
                    fontdict=dict(fontsize=12, color='r',
                                  family='monospace'),
                    bbox={'facecolor': '#74C476',
                          'edgecolor': 'b', 'alpha': 0.5, 'pad': 8}
                    )
    text.set_color('b')
    plt.savefig(f"{save_dir}/domain_predict.png", dpi=100, bbox_inches='tight')


class TestMaxwell3DSlab():
    """
    Test the model.
    """

    def __init__(self, config):
        self.config = config

        self.net = self.init_net()
        self.hessian_ex_xx = SecondOrderGrad(self.net, 0, 0, output_idx=0)
        self.hessian_ex_yy = SecondOrderGrad(self.net, 1, 1, output_idx=0)
        self.hessian_ex_zz = SecondOrderGrad(self.net, 2, 2, output_idx=0)

        self.hessian_ey_xx = SecondOrderGrad(self.net, 0, 0, output_idx=1)
        self.hessian_ey_yy = SecondOrderGrad(self.net, 1, 1, output_idx=1)
        self.hessian_ey_zz = SecondOrderGrad(self.net, 2, 2, output_idx=1)

        self.hessian_ez_xx = SecondOrderGrad(self.net, 0, 0, output_idx=2)
        self.hessian_ez_yy = SecondOrderGrad(self.net, 1, 1, output_idx=2)
        self.hessian_ez_zz = SecondOrderGrad(self.net, 2, 2, output_idx=2)

        self.reshape = ops.Reshape()
        self.concat = ops.Concat(1)
        self.abs = ops.Abs()

        self.zeros_like = ops.ZerosLike()
        self.reduce_mean = ops.ReduceMean()

        self.wave_number = Tensor(
            self.config["wave_number"], ms.dtype.float32)
        self.pi = Tensor(PI, ms.dtype.float32)

        self.xyrange = (self.config["coord_min"][0],
                        self.config["coord_max"][0])

    def init_net(self):
        """
        Initialize the network and load weight.
        """

        def load_paramters_into_net(param_path, net):
            param_dict = load_checkpoint(param_path)
            convert_ckpt_dict = {}
            for _, param in net.parameters_and_names():
                convert_name1 = "jac2.model.model.cell_list." + param.name
                convert_name2 = "jac2.model.model.cell_list." + \
                                ".".join(param.name.split(".")[2:])
                for key in [convert_name1, convert_name2]:
                    if key in param_dict:
                        convert_ckpt_dict[param.name] = param_dict[key]
            load_param_into_net(net, convert_ckpt_dict)
            print("Load parameters finished!")

        net = MultiScaleFCCell(in_channel=self.config["in_channel"],
                               out_channel=self.config["out_channel"],
                               layers=self.config["layers"],
                               neurons=self.config["neurons"],
                               )
        load_paramters_into_net(self.config["param_path"], net)
        return net

    def run(self):
        """
        Create a cubic grid data to evaluate the network.
        """
        print("<===================== Begin evaluating =====================>")
        t_start = time.time()
        xmin, ymin, _ = self.config["coord_min"]
        xmax, ymax, _ = self.config["coord_max"]
        xyrange = (xmin, xmax)
        axis_size = self.config["axis_size"]
        save_dir = self.config["result_save_dir"]
        data_path = self.config["waveguide_points_path"]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_dir.endswith('/'):
            save_dir = save_dir[:-1]

        u = np.linspace(xmin, xmax, axis_size)
        v = np.linspace(ymin, ymax, axis_size)
        ugrid, vgrid = np.meshgrid(u, v)
        uu, vv = ugrid.reshape(-1, 1), vgrid.reshape(-1, 1)
        ones = np.ones_like(uu)
        data = np.load(data_path)  # load data from Nvidia Modulus
        # The plane to detect
        plane1 = np.c_[ones * 0, uu, vv]  # x=0, yz,
        plane2 = np.c_[uu, ones * 0, vv]  # xz, y=0
        plane3 = np.c_[uu, vv, ones * 0]  # xy, z=0

        # plane0 is the true-value on waveguide-port plane.
        # data.shape=(13277, 9), the first columns are (x,y), others are
        # Ez on different conditions.
        plane0 = np.c_[-0.5 * np.ones(len(data[:, 0])), data[:, 0], data[:, 1]]
        e0z_true = data[:, 2]
        e0 = self.net(ms.Tensor(plane0, ms.dtype.float32)
                      ).asnumpy()
        e0z_pred = e0[:, 2]
        e0z_diff = e0z_pred - e0z_true
        e0x_diff = e0[:, 0]
        e0y_diff = e0[:, 1]

        print(f"Max difference of waveguide port in Ex: {e0x_diff.max():.5f}")
        print(f"Max difference of waveguide port in Ey: {e0y_diff.max():.5f}")
        print(f"Max difference of waveguide port in Ez: {e0z_diff.max():.5f}")
        plot_waveguide(data[:, 0], data[:, 1], e0z_pred, e0z_true, e0z_diff, e0x_diff, e0y_diff, save_dir)
        print("plot waveguide completed!")

        # Plane 1,2,3
        e1 = self.net(ms.Tensor(plane1, ms.dtype.float32)
                      ).asnumpy()
        e2 = self.net(ms.Tensor(plane2, ms.dtype.float32)
                      ).asnumpy()
        e3 = self.net(ms.Tensor(plane3, ms.dtype.float32)
                      ).asnumpy()

        e1 = e1.reshape((ugrid.shape[0], ugrid.shape[1], 3)).transpose(2, 0, 1)
        e2 = e2.reshape((ugrid.shape[0], ugrid.shape[1], 3)).transpose(2, 0, 1)
        e3 = e3.reshape((ugrid.shape[0], ugrid.shape[1], 3)).transpose(2, 0, 1)
        plot_domain_plane(e1, e2, e3, xyrange, save_dir)
        print("plot domain result completed!")

        print("Begin scan the whole volumn, it may take a long time.")
        # result[i, x, y, z], i=0 -> Ex,  i=1 -> Ey, i=2 -> Ez,
        # (x,y,z) is the 3-D coordinates.
        result = np.zeros(shape=(3, axis_size, axis_size, axis_size), dtype=np.float32)
        for i, x in tqdm.tqdm(enumerate(np.linspace(xmin, xmax, axis_size))):
            xx = ones * x
            points = ms.Tensor(np.c_[xx, uu, vv], ms.dtype.float32)
            u_xyz = self.net(points).asnumpy()
            result[0, i, :, :] = u_xyz[:, 0].reshape((axis_size, axis_size))
            result[1, i, :, :] = u_xyz[:, 1].reshape((axis_size, axis_size))
            result[2, i, :, :] = u_xyz[:, 2].reshape((axis_size, axis_size))
        np.save(f"{save_dir}/slab_result.npy", result)

        print("<===================== End evaluating =====================>")
        t_end = time.time()
        print(
            f"This evaluation total spend {(t_end - t_start) / 60:.2f} minutes.")


if __name__ == "__main__":
    config_ = load_config()
    evaluation(config_)
