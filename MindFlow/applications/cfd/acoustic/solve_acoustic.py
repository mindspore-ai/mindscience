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
""""Solve 2D acoustic equation"""""
import os
import argparse
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import mindspore as ms
from mindspore import ops, Tensor, numpy as mnp

from mindflow.utils import load_yaml_config

from cbs.cbs import CBS
from src import utils, visual


def solve_cbs(cbs, velo, slocs, omegas, receiver_zs=None, dx=1., n_batches=1):
    '''
    Solve for different source locations and frequencies using CBS (Convergent Born series) solver
    Args:
        velo: 2d Tensor, the velocity field
        slocs: (ns, 2) array, the source locations (z, x coordinates) to be solved
        omegas: 1d array, the frequencies to be solved on
        receiver_zs: 1d array, z coordinates of signal receivers.
            Default is None, which means all signals will be received
        dx: float, the grid interval along x & z directions
        n_batches: int, the number of batches for frequencies to be diveded into
    Returns:
        u_real, u_imag:
    '''
    no = len(omegas)
    ns = len(slocs)
    nz, nx = velo.shape

    if receiver_zs is None:
        receiver_zs = np.arange(nz) * dx

    krs = Tensor(np.rint(np.divide(receiver_zs, dx)), dtype=ms.int32, const_arg=False)
    omegas = Tensor(omegas, dtype=ms.float32, const_arg=False)

    masks = Tensor(utils.sloc2mask(slocs, (nz, nx), (dx, dx)), dtype=ms.float32, const_arg=False) # shape (ns, nz, nx)

    urs = [] # note: do hold the solution of each batch in list and cat to Tensor later
    uis = [] # note: do not hold them by modifying Tensor slices, dynamic shape and error would be caused
    errs = []

    for n, i in enumerate(range(0, no, no // n_batches)):
        j = i + min(no // n_batches, no - i)

        print(f'batch {n}, omega {float(omegas[i]):.4f} ~ {float(omegas[j-1]):.4f}')

        c_star = velo / dx / omegas[i:j].reshape(-1, 1, 1)
        f_star = masks.reshape(ns, 1, nz, nx)
        c_star, f_star = mnp.broadcast_arrays(c_star, f_star)

        c_star = c_star.reshape(-1, 1, *c_star.shape[2:]) # shape (ns * no, 1, nz, nx)
        f_star = f_star.reshape(-1, 1, *f_star.shape[2:]) # shape (ns * no, 1, nz, nx)

        ur, ui, err = cbs.solve(c_star, f_star, tol=1e-3)

        urs.append(ur[..., krs, :].reshape(ns, -1, len(krs), nx))
        uis.append(ui[..., krs, :].reshape(ns, -1, len(krs), nx))
        errs.append(np.reshape(err, (-1, ns, j - i)))

    u_real = ops.cat(urs, axis=1) # shape (ns, no, len(krs), nx)
    u_imag = ops.cat(uis, axis=1) # shape (ns, no, len(krs), nx)

    return u_real, u_imag, errs


def main(config):
    data_config = config['data']
    solve_config = config['solve']
    summary_config = config['summary']

    # read time & frequency points
    dt = solve_config['dt']
    nt = solve_config['nt']
    ts = np.arange(nt) * dt
    omegas_all = np.fft.rfftfreq(nt) * (2 * np.pi / dt)

    # read source locations
    df = pd.read_csv(os.path.join(data_config['root_dir'], data_config['source_locations']), index_col=0)
    slocs = df[['y', 'x']].values # shape (ns, 2)

    # read & interp source wave
    df = pd.read_csv(os.path.join(data_config['root_dir'], data_config['source_wave']))
    inter_func = interp1d(df.t, df.f, bounds_error=False, fill_value=0)
    src_waves = inter_func(ts) # shape (nt)
    src_amplitudes = np.fft.rfft(src_waves) # shape (nt//2+1)

    # read velocity array
    velo = np.load(os.path.join(data_config['root_dir'], data_config['velocity_field']))
    nz, nx = velo.shape
    dx = data_config['velocity_dx']

    # select omegas
    no = len(omegas_all) // solve_config['downsample_rate']

    if solve_config['downsample_mode'] == 'exp':
        omegas_sel = np.exp(np.linspace(np.log(omegas_all[1]), np.log(omegas_all[-1]), no))
    elif solve_config['downsample_mode'] == 'square':
        omegas_sel = np.linspace(omegas_all[1]**.5, omegas_all[-1]**.5, no)**2
    else:
        omegas_sel = np.linspace(omegas_all[1], omegas_all[-1], no)

    # send to NPU and perform computation
    os.makedirs(summary_config['root_dir'], exist_ok=True)
    velo = Tensor(velo, dtype=ms.float32, const_arg=True)
    cbs = CBS((nz, nx), remove_pml=False)

    ur, ui, errs = solve_cbs(
        cbs, velo, slocs, omegas_sel, dx=dx, n_batches=solve_config['n_batches']) # shape (ns, no, len(receiver_zs), nx)

    u_star = np.squeeze(ur.numpy() + 1j * ui.numpy()) # shape (ns, no, len(krs), nx)
    np.save(os.path.join(summary_config['root_dir'], 'u_star.npy'), u_star)

    # recover dimension and interpolate to full frequency domain
    u_star /= omegas_sel.reshape(-1, 1, 1)**2
    u_star = interp1d(omegas_sel, u_star, axis=1, kind='cubic', bounds_error=False, fill_value=0)(omegas_all)
    u_star *= src_amplitudes.reshape(-1, 1, 1)

    # transform to time domain
    u_time = np.fft.irfft(u_star, axis=1)
    np.save(os.path.join(summary_config['root_dir'], 'u_time.npy'), u_time)

    # visualize the result
    u_time = np.load(os.path.join(summary_config['root_dir'], 'u_time.npy'))
    visual.anim(velo.numpy(), u_time, ts, os.path.join(summary_config['root_dir'], 'wave.gif'))
    visual.plot_errs(errs, os.path.join(summary_config['root_dir'], 'errors.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve 2D acoustic equation with CBS")
    parser.add_argument(
        "--mode",
        type=str,
        default="GRAPH",
        choices=["GRAPH", "PYNATIVE"],
        help="Running in GRAPH_MODE OR PYNATIVE_MODE",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=utils.choose_free_npu(),
        help="ID of the target device",
    )
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    ms.set_context(
        device_target='Ascend',
        device_id=args.device_id,
        mode=ms.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else ms.PYNATIVE_MODE)

    main(load_yaml_config(args.config_file_path))
