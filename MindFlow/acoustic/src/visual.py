# Copyright 2025 Huawei Technologies Co., Ltd
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
"""visualize the results and computing history"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def anim(velo, us, ts, figname):
    '''
    Animate the wave field in time domain and generate gif
    '''
    ns, nt, _, _ = us.shape

    nrows = 1
    ncols = ns + 1

    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, squeeze=False,
        constrained_layout=True, figsize=(3 * ncols, 3 * nrows))

    axs[0, 0].contourf(velo, cmap='seismic', extend='both')
    axs[0, 0].set_title('velocity')

    for ax in axs.ravel():
        ax.yaxis.set_inverted(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

    for i, ax in enumerate(axs[0, 1:]):
        ax.set_title(f'shot_{i}')
    for ax in axs[-1]:
        ax.set_xlabel('x')
    for ax in axs[:, 0]:
        ax.set_ylabel('z')

    handles = []

    def run(n):
        print(f'animating wave field {n} / {nt}')

        while handles:
            handles.pop().remove()

        for j, u in enumerate(us):
            ax = axs[0, j + 1]
            cnt = ax.contourf(u[n], cmap='bwr', extend='both', levels=np.linspace(-3, 3, 10) * u.std())
            handles.append(cnt)
            ax.set_title(f"t = {ts[n]:.3f} [s]")

    ani = animation.FuncAnimation(fig, run, frames=range(0, nt, 5), interval=100, repeat_delay=1000, repeat=True)

    ani.save(figname, dpi=100)
    plt.close()

def plot_errs(errs, figname):
    '''
    Plot the convergence history
    Args:
        errs: list of arrays, each array has shape (iters, ns, no_batchsize)
    '''
    fig, ax = plt.subplots(constrained_layout=True)
    for err in errs:
        data = np.mean(err, axis=1) # average out the slocs dimension
        ax.semilogy(data)
    ax.grid()
    ax.legend(['different frequencies'])
    ax.set_xlabel('iteration')
    ax.set_ylabel('residual')
    fig.savefig(figname, dpi=300)
    plt.close()
