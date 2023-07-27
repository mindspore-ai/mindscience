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
from src.config import maxwell_3d_config


def load_config():
    """load config"""
    config = copy.deepcopy(maxwell_3d_config)
    return config


def evaluation(config):
    """evaluation"""
    tester = TestMaxwell3DCavity(config=config)
    tester.run()


def plot_waveguide(ez_pred, ez_true, ez_diff, xyrange=(0, 2), save_dir=""):
    """
    Visualize the waveguide port.

    Args:
        ez_pred: Ez prediction.
        ez_true: Ez ground-truth.
        ez_diff: Ez error.
        xyrange: axis range.
        save_dir: the directory to save the result.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    ax0 = axes[0].imshow(ez_pred, vmin=0, vmax=1.0, cmap='jet')
    axes[0].set_xticks([0, len(ez_pred)], xyrange)
    axes[0].set_yticks([0, len(ez_pred)], xyrange)
    axes[0].set_title("ground truth")

    axes[1].imshow(ez_true, vmin=0, vmax=1.0, cmap='jet')
    axes[1].set_xticks([0, len(ez_pred)], xyrange)
    axes[1].set_yticks([0, len(ez_true)], xyrange)
    axes[1].set_title("prediction")

    vmax = np.ceil(ez_diff.max() * 100) / 100
    ax2 = axes[2].imshow(ez_diff, vmin=0, vmax=vmax, cmap='binary')
    axes[2].set_xticks([0, len(ez_pred)], xyrange)
    axes[2].set_yticks([0, len(ez_diff)], xyrange)
    axes[2].set_title("difference")

    fig.colorbar(ax0, ax=[axes[0], axes[1]], shrink=0.8)
    fig.colorbar(ax2, ax=axes[2], shrink=0.8)

    plt.savefig(f"{save_dir}/waveguide_Ez.png", dpi=200, bbox_inches='tight')


def plot_domain_result(u1, u2, u3, xyrange, save_dir=""):
    """
    Visualize the domain result by 3 specific plane (u1, u2, u3).

    Args:
        u1, u2, u3: The data need to be visualizeed.
        xyrange:    Axis range.
        save_dir:   Save directory.
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
                    s='Top to down is 3 planes: x=1, y=1, z=1\nLeft to right are 3 components: $E_x$, $E_y$, $E_z$',
                    fontdict=dict(fontsize=12, color='r', family='monospace'),
                    bbox={'facecolor': '#74C476',
                          'edgecolor': 'b', 'alpha': 0.5, 'pad': 8}
                    )
    text.set_color('b')
    plt.savefig(f"{save_dir}/domain_predict.png", dpi=200, bbox_inches='tight')


class TestMaxwell3DCavity():
    """
    Test the model.
    """

    def __init__(self, config):
        self.config = config
        self.net = self.init_net()
        self.concat = ops.Concat(1)
        self.abs = ops.Abs()
        self.zeros_like = ops.ZerosLike()
        self.pi = Tensor(PI, ms.dtype.float32)
        self.wave_number = Tensor(self.config["wave_number"], ms.dtype.float32)
        self.eigenmode = Tensor(self.config["eigenmode"], ms.dtype.float32)
        self.xyrange = (self.config["coord_min"][0],
                        self.config["coord_max"][0])

    def init_net(self):
        """
        Initialize the network.
        """

        def load_paramters_into_net(param_path, net):
            """
            Load parameter into net.
            """
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
        Create some data to evaluate the network.
        """

        print("<===================== Begin evaluating =====================>")
        t_start = time.time()
        xmin, ymin, dummy_zmin = self.config["coord_min"]
        xmax, ymax, dummy_zmax = self.config["coord_max"]
        xyrange = (xmin, xmax)
        axis_size = self.config["axis_size"]
        save_dir = self.config["result_save_dir"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_dir.endswith('/'):
            save_dir = save_dir[:-1]

        u = np.linspace(xmin, xmax, axis_size)
        v = np.linspace(ymin, ymax, axis_size)
        ugrid, vgrid = np.meshgrid(u, v)
        uu, vv = ugrid.reshape(-1, 1), vgrid.reshape(-1, 1)
        ones = np.ones_like(uu)

        # Self-define 4 plane to visualize the result.
        plane0 = np.c_[ones * 0, uu, vv]
        plane1 = np.c_[ones, uu, vv]
        plane2 = np.c_[uu, ones, vv]
        plane3 = np.c_[uu, vv, ones]

        # Plane 0 is the waveguide port plane.
        label0, e0, diff0 = self.get_waveguide_residual(plane0)
        shape = ugrid.shape
        ez_pred = e0[:, 2].reshape(shape)
        ez_true = label0[:, 2].reshape(shape)
        ez_diff = diff0[:, 2].reshape(shape)

        print(
            f"Max difference of waveguide port in Ex: {diff0[:, 0].max():.5f}")
        print(
            f"Max difference of waveguide port in Ey: {diff0[:, 1].max():.5f}")
        print(
            f"Max difference of waveguide port in Ez: {diff0[:, 2].max():.5f}")
        plot_waveguide(ez_pred, ez_true, ez_diff, xyrange, save_dir)
        print("plot waveguide completed!")

        # Plane 1, 2, 3
        e1 = self.net(ms.Tensor(plane1, ms.dtype.float32)
                      ).asnumpy()
        e2 = self.net(ms.Tensor(plane2, ms.dtype.float32)
                      ).asnumpy()
        e3 = self.net(ms.Tensor(plane3, ms.dtype.float32)
                      ).asnumpy()

        e1 = e1.reshape((ugrid.shape[0], ugrid.shape[1], 3)).transpose(2, 0, 1)
        e2 = e2.reshape((ugrid.shape[0], ugrid.shape[1], 3)).transpose(2, 0, 1)
        e3 = e3.reshape((ugrid.shape[0], ugrid.shape[1], 3)).transpose(2, 0, 1)
        plot_domain_result(e1, e2, e3, xyrange, save_dir)
        print("plot domain result completed!")

        print("Begin scan the whole volumn, it may take a long time.")
        # result[i, x, y, z]. i=0 -> Ex,  i=1 -> Ey, i=2 -> Ez
        # (x,y,z) is the 3-D coordinate.
        result = np.zeros(shape=(3, axis_size, axis_size, axis_size), dtype=np.float32)
        for i, x in tqdm.tqdm(enumerate(np.linspace(xmin, xmax, axis_size))):
            xx = ones * x
            points = ms.Tensor(np.c_[xx, uu, vv], ms.dtype.float32)
            u_xyz = self.net(points).asnumpy()
            result[0, i, :, :] = u_xyz[:, 0].reshape((axis_size, axis_size)).T
            result[1, i, :, :] = u_xyz[:, 1].reshape((axis_size, axis_size)).T
            result[2, i, :, :] = u_xyz[:, 2].reshape((axis_size, axis_size)).T
        np.save(f"{save_dir}/cavity_result.npy", result)

        print("<===================== End evaluating =====================>")
        t_end = time.time()
        print(
            f"This evaluation total spend {(t_end - t_start) / 60:.2f} minutes.")

    def get_waveguide_residual(self, data):
        """
        Calculate the boundary error(residual).

        Args:
            data: shape=(n,3), n is the number of sampling points

        Return:
            label: shape=(n,3), ground-truth
            u: shape=(n,3), prediction of (Ex, Ey, Ez)
            diff: shape=(n,3), difference of abs(label-u)
        """
        data = ms.Tensor(data, ms.dtype.float32)
        u = self.net(data)
        # Ground-truth: Ez = sin(m * pi * y / height) * sin(m * pi * y / length)
        height = self.config["coord_max"][1]
        length = self.config["coord_max"][2]
        # data[:,0]->x, data[:,1]->y, data[:,2]->z
        label_z = ops.sin(self.eigenmode * self.pi * data[:, 1:2] / height) * \
                  ops.sin(self.eigenmode * self.pi * data[:, 2:3] / length)
        label = self.concat(
            (self.zeros_like(label_z), self.zeros_like(label_z), label_z))
        diff = self.abs(u - label)
        return label.asnumpy(), u.asnumpy(), diff.asnumpy()


if __name__ == "__main__":
    config_ = load_config()
    evaluation(config_)
