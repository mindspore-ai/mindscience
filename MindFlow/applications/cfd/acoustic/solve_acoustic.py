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
""""Solve 2D/3D acoustic equation"""""
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


def solve_cbs(cbs, velo, slocs, omegas, receivers=None, dxs=None, n_batches=1):
    '''
    Solve for different source locations and frequencies using CBS (Convergent Born series) solver
    Args:
        velo: 2d/3d Tensor, the velocity field.
        slocs: (ns, 2) or (ns, 3) array, the source locations (z, x) or (z, y, x) coordinates
            to be solved.
        omegas: 1d array, the frequencies to be solved on.
        receivers: Tuple of 1d array, (z, x) or (z, y, x) coordinates of signal receivers.
            Default is None, which means all signals will be received.
        dxs: Tuple of float, the grid interval along z & x (2D) or z & y & x (3D) directions.
            Default is None.
        n_batches: int, the number of batches for frequencies to be diveded into.
    Returns:
        u_real, u_imag:
    '''
    no = len(omegas)
    ns = len(slocs)
    dim = len(velo.shape)
    if dim == 2:
        nz, nx = velo.shape
    else:
        nz, ny, nx = velo.shape
    if dxs is None:
        dxs = (1.0,) * dim

    if receivers is None:
        if dim == 2:
            receivers = (np.arange(nz) * dxs[0], np.arange(nx) * dxs[1])
        else:
            receivers = (np.arange(nz) * dxs[0], np.arange(ny) * dxs[1], np.arange(nx) * dxs[2])

    if dim == 2:
        krzs = Tensor(np.rint(np.divide(receivers[0], dxs[0])), dtype=ms.int32, const_arg=False)
        krxs = Tensor(np.rint(np.divide(receivers[1], dxs[1])), dtype=ms.int32, const_arg=False)
    else:
        krzs = Tensor(np.rint(np.divide(receivers[0], dxs[0])), dtype=ms.int32, const_arg=False)
        krys = Tensor(np.rint(np.divide(receivers[1], dxs[1])), dtype=ms.int32, const_arg=False)
        krxs = Tensor(np.rint(np.divide(receivers[2], dxs[2])), dtype=ms.int32, const_arg=False)

    omegas = Tensor(omegas, dtype=ms.float32, const_arg=False)

    # Shape (ns, nz, nx) for 2D and (ns, nz, ny, nx) for 3D
    masks = Tensor(utils.sloc2mask(slocs, velo.shape, dxs), dtype=ms.float32, const_arg=False)

    urs = [] # Note: do hold the solution of each batch in list and cat to Tensor later
    uis = [] # Note: do not hold them by modifying Tensor slices, dynamic shape and error would be caused
    errs = []

    for n, i in enumerate(range(0, no, no // n_batches)):
        j = i + min(no // n_batches, no - i)

        print(f'batch {n}, omega {float(omegas[i]):.4f} ~ {float(omegas[j-1]):.4f}')

        if dim == 2:
            c_star = velo / dxs[-1] / omegas[i:j].reshape(-1, 1, 1)
        else:
            c_star = velo / dxs[-1] / omegas[i:j].reshape(-1, 1, 1, 1)

        f_star = masks.reshape(ns, 1, *velo.shape)
        c_star, f_star = mnp.broadcast_arrays(c_star, f_star)

        # Shape (ns * no, 1, nz, nx) for 2D and (ns * no, 1, nz, ny, nx) for 3D
        c_star = c_star.reshape(-1, 1, *c_star.shape[2:])
        f_star = f_star.reshape(-1, 1, *f_star.shape[2:])

        ur, ui, err = cbs.solve(c_star, f_star, tol=1e-3)

        if dim == 2:
            krzs_expand = krzs.reshape(-1, 1)
            krxs_expand = krxs.reshape(1, -1)
            urs.append(ur[..., krzs_expand, krxs_expand].reshape(ns, -1, len(krzs), len(krxs)))
            uis.append(ui[..., krzs_expand, krxs_expand].reshape(ns, -1, len(krzs), len(krxs)))
        else:
            krzs_expand = krzs.reshape(-1, 1, 1)
            krys_expand = krys.reshape(1, -1, 1)
            krxs_expand = krxs.reshape(1, 1, -1)
            urs.append(ur[..., krzs_expand, krys_expand, krxs_expand].reshape(
                ns, -1, len(krzs), len(krys), len(krxs)))
            uis.append(ui[..., krzs_expand, krys_expand, krxs_expand].reshape(
                ns, -1, len(krzs), len(krys), len(krxs)))

        errs.append(np.reshape(err, (-1, ns, j - i)))

    u_real = ops.cat(urs, axis=1) # Shape (ns, no, len(krs), nx)
    u_imag = ops.cat(uis, axis=1) # Shape (ns, no, len(krs), nx)

    return u_real, u_imag, errs


def main(dim):
    if dim == 2:
        config = load_yaml_config("./config_2d.yaml")
    elif dim == 3:
        config = load_yaml_config("./config_3d.yaml")
    else:
        raise ValueError("The dim can only choose 2 or 3.")

    data_config = config['data']
    solve_config = config['solve']
    summary_config = config['summary']

    # Coarsen rate for reducing scale
    rate = solve_config['coarsen_rate']

    # Read time & frequency points
    dt = solve_config['dt']
    nt = solve_config['nt']
    dt *= rate # Increase the time iteraion step size
    nt //= rate
    receivers = None

    # Read velocity array
    velo = np.load(os.path.join(data_config['root_dir'], data_config['velocity_field']))
    if dim != len(velo.shape):
        raise ValueError("The dim and dimension of velocity should be equal.")
    dx = data_config['velocity_dx']
    dz = data_config['velocity_dz']
    if dim == 2:
        dxs = (rate*dz, rate*dx)
        velo = velo[::rate, ::rate] # Coarsen velocity field
    else:
        dy = data_config['velocity_dy']
        dxs = (rate*dz, rate*dy, rate*dx)
        velo = velo[::rate, ::rate, ::rate] # Coarsen velocity field

    # Read source locations
    df = pd.read_csv(os.path.join(data_config['root_dir'], data_config['source_locations']), index_col=0)
    if dim == 2:
        slocs = df[['y', 'x']].values # Shape (ns, 2)
    else:
        slocs = df[['z', 'y', 'x']].values # Shape (ns, 3)

    # Read & interp source wave
    df = pd.read_csv(os.path.join(data_config['root_dir'], data_config['source_wave']))
    inter_func = interp1d(df.t, df.f, kind='cubic', bounds_error=False, fill_value=0)

    ts = np.arange(nt) * dt
    omegas_all = np.fft.rfftfreq(nt) * (2 * np.pi / dt)

    # Interpolation source wave
    src_waves = inter_func(ts) # Shape (nt)
    src_amplitudes = np.fft.rfft(src_waves) # Shape (nt//2+1)

    # Select omegas
    no = len(omegas_all) // solve_config['downsample_rate']

    if solve_config['downsample_mode'] == 'exp':
        omegas_sel = np.exp(np.linspace(np.log(omegas_all[1]), np.log(omegas_all[-1]), no))
    elif solve_config['downsample_mode'] == 'square':
        omegas_sel = np.linspace(omegas_all[1]**.5, omegas_all[-1]**.5, no)**2
    else:
        omegas_sel = np.linspace(omegas_all[1], omegas_all[-1], no)

    # Send to NPU and perform computation
    os.makedirs(summary_config['root_dir'], exist_ok=True)
    velo = Tensor(velo, dtype=ms.float32, const_arg=True)

    # Solve Helmholtz equations with CBS
    dxs_nd = tuple(d / dxs[-1] for d in dxs) # Nondimensional dxs
    pml_size = tuple(solve_config['pml_size'])
    btype = solve_config['btype']
    cbs = CBS(velo.shape, dxs=dxs_nd, pml_size=pml_size, alpha=1., rampup=12,
              btype=btype, remove_pml=False)

    ur, ui, errs = solve_cbs(
        cbs, velo, slocs, omegas_sel, receivers=receivers, dxs=dxs, n_batches=solve_config['n_batches'])

    u_star = ur.numpy() + 1j * ui.numpy() # Shape (ns, no, len(krzs), len(krys), len(krxs))

    np.save(os.path.join(summary_config['root_dir'], 'u_star.npy'), np.squeeze(u_star))

    # Recover dimension and interpolate to full frequency domain
    ones = (1,) * dim
    u_star /= omegas_sel.reshape(-1, *ones)**2
    u_star = interp1d(omegas_sel, u_star, axis=1, kind='cubic', bounds_error=False, fill_value=0)(omegas_all)
    u_star *= src_amplitudes.reshape(-1, *ones)

    # Transform to time domain
    u_time = np.fft.irfft(u_star, axis=1)
    np.save(os.path.join(summary_config['root_dir'], 'u_time.npy'), u_time)

    # Visualize the result
    u_time = np.load(os.path.join(summary_config['root_dir'], 'u_time.npy'))
    if dim == 2:
        visual.anim(velo.numpy(), u_time, ts, os.path.join(summary_config['root_dir'], 'wave.gif'))
    else:
        visual.anim3d(velo.numpy(), u_time, ts, os.path.join(summary_config['root_dir'], 'wave.gif'))
    visual.plot_errs(errs, os.path.join(summary_config['root_dir'], 'errors.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve 2D/3D acoustic equation with CBS")
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
    parser.add_argument(
        "--dim",
        type=int,
        default=3,
        help="Dimension of acoustic equation"
    )
    args = parser.parse_args()

    ms.set_context(
        device_target='Ascend',
        device_id=args.device_id,
        mode=ms.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else ms.PYNATIVE_MODE)

    main(args.dim)
