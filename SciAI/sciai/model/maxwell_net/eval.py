
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

"""maxwell net eval"""
import os

import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sciai.context import init_project
from sciai.utils import print_log, amp2datatype
from sciai.utils.python_utils import print_time
from src.network import MaxwellNet, LossNet
from src.process import load_data, prepare


def evaluation(args, model, ri_ms, scat_pot_ms):
    """evaluation"""
    loss_cell = LossNet(model)
    loss = loss_cell(scat_pot_ms, ri_ms)
    print_log("loss: ", loss)
    if not args.save_fig:
        return
    diff, total = model(scat_pot_ms, ri_ms)
    total_np, diff_np = total.asnumpy(), diff.asnumpy()
    total_np = total_np[0, 0::2, :, :] + 1j * total_np[0, 1::2, :, :]
    diff_np = diff_np[0, 0::2, :, :] + 1j * diff_np[0, 1::2, :, :]
    scat_pot_np, ri = scat_pot_ms.asnumpy(), ri_ms.asnumpy()
    scat_pot_np = scat_pot_np[0, :, :, :] * (ri - 1) + 1
    # Min max values to present the data just for visualization is approximately correct.
    # The output values from MaxwellNet are defined on Yee grid.
    # So, if you want to quantitatively compare the MaxwellNet outputs with solutions from another solver,
    # you should compare two solutions at the same Yee grid points.
    nx, nz = args.nx * (args.symmetry_x + 1), args.nz
    delta = args.wavelength / args.dpl
    x_min, x_max = -(nx // 2) * delta, (nx // 2 - 1) * delta
    z_min, z_max = -(nz // 2) * delta, (nz // 2 - 1) * delta
    xz = x_max, x_min, z_max, z_min
    if args.problem == 'te':
        if args.symmetry_x:
            scat_pot_np = np.pad(np.concatenate(
                (np.flip(scat_pot_np[0, 1::, :], 0), scat_pot_np[0, :, :]), 0), ((1, 0), (0, 0)))
            total_np = np.pad(np.concatenate((np.flip(total_np[0, 1::, :], 0), total_np[0, :, :]), 0), ((1, 0), (0, 0)))
            diff_np = np.pad(np.concatenate((np.flip(diff_np[0, 1::, :], 0), diff_np[0, :, :]), 0), ((1, 0), (0, 0)))

        plot_te(args, ri, scat_pot_np, total_np, xz)

    elif args.problem == 'tm':
        if args.symmetry_x:
            scat_pot_x_np = np.concatenate((np.flip(scat_pot_np[0, :, :], 0), scat_pot_np[0, :, :]), 0)
            scat_pot_z_np = np.pad(np.concatenate(
                (np.flip(scat_pot_np[1, 1::, :], 0), scat_pot_np[1, :, :]), 0), ((1, 0), (0, 0)))
            total_x_np = np.concatenate((np.flip(total_np[0, :, :], 0), total_np[0, :, :]), 0)
            total_z_np = np.pad(np.concatenate(
                (-np.flip(total_np[1, 1::, :], 0), total_np[1, :, :]), 0), ((1, 0), (0, 0)))
        else:
            raise ValueError("symmetry_x must be True in tm")

        plot_tm(args, ri, (scat_pot_x_np, scat_pot_z_np), (total_x_np, total_z_np), xz)
    else:
        raise KeyError("'mode' should me either 'te' or 'tm'.")


def plot_te(args, ri, scat_pot_np, total_np, xz):
    """plot te"""
    fontsize = 20
    x_max, x_min, z_max, z_min = xz
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle('TE mode - Sherical Lens', fontsize=fontsize)
    img0 = axs[0].imshow(scat_pot_np, extent=[z_min, z_max, x_min, x_max], vmin=1, vmax=ri)
    axs[0].set_title('RI distribution', fontsize=fontsize)
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img0, cax=cax0)
    img1 = axs[1].imshow(np.abs(total_np), extent=[z_min, z_max, x_min, x_max])
    axs[1].set_title('Ey (envelop)', fontsize=fontsize)
    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_path, 'te_result.png'))


def plot_tm(args, ri, scat_pot_np, total_np, xz):
    """plot tm"""
    fontsize = 20
    scat_pot_x_np, scat_pot_z_np = scat_pot_np
    total_x_np, total_z_np = total_np
    x_max, x_min, z_max, z_min = xz
    fig, axs = plt.subplots(2, 2, figsize=(8, 10))
    fig.suptitle('TM mode - Sherical Lens', fontsize=fontsize)
    img00 = axs[0, 0].imshow(scat_pot_x_np, extent=[z_min, z_max, x_min, x_max], vmin=1, vmax=ri)
    axs[0, 0].set_title('RI distribution', fontsize=fontsize)
    divider00 = make_axes_locatable(axs[0, 0])
    cax00 = divider00.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img00, cax=cax00)
    img01 = axs[0, 1].imshow(np.abs(total_x_np), extent=[z_min, z_max, x_min, x_max])
    axs[0, 1].set_title('Ex (envelop)', fontsize=fontsize)
    divider01 = make_axes_locatable(axs[0, 1])
    cax01 = divider01.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img01, cax=cax01)
    img10 = axs[1, 0].imshow(scat_pot_z_np, extent=[z_min, z_max, x_min, x_max], vmin=1, vmax=ri)
    axs[1, 0].set_title('RI distribution', fontsize=fontsize)
    divider10 = make_axes_locatable(axs[1, 0])
    cax10 = divider10.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img10, cax=cax10)
    img11 = axs[1, 1].imshow(np.abs(total_z_np), extent=[z_min, z_max, x_min, x_max])
    axs[1, 1].set_title('Ez (envelop)', fontsize=fontsize)
    divider11 = make_axes_locatable(axs[1, 1])
    cax11 = divider11.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img11, cax=cax11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_path, 'tm_result.png'))


@print_time("eval")
def main(args):
    """main"""
    dtype = amp2datatype(args.amp_level)
    scat_pot_ms, ri_ms = load_data(args, dtype)
    model = MaxwellNet(args)
    ms.load_checkpoint(args.load_ckpt_path, model)
    if dtype == ms.float16:
        model.to_float(ms.float16)
    evaluation(args, model, ri_ms, scat_pot_ms)


if __name__ == '__main__':
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)
