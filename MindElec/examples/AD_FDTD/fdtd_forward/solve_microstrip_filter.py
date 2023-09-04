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
"""
forward problem solve process
"""
import argparse
import os

import numpy as np
from mindspore import context

from src import Antenna, SParameterSolver
from src import CFSParameters, Gaussian
from src import GridHelper, UniformBrick, PECPlate, VoltageSource, Resistor
from src import VoltageMonitor, CurrentMonitor
from src import estimate_time_interval, compare_s
from src import full3d


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(
        description='FDTD-Based Electromagnetics Forward-Problem Solver')
    parser.add_argument('--device_target', type=str, default=None)
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--nt', type=int, default=3000,
                        help='Number of time steps.')
    parser.add_argument('--max_call_depth', type=int, default=1000)
    parser.add_argument('--dataset_dir', type=str,
                        default='./dataset', help='dataset directory')
    parser.add_argument('--result_dir', type=str,
                        default='./result', help='result directory')
    parser.add_argument('--cfl_number', type=float, default=0.9, help='CFL number')
    parser.add_argument('--fmax', type=float, default=20e9,
                        help='highest frequency (Hz)')
    options = parser.parse_args()
    return options


def get_waveform_t(nt, dt, fmax):
    """
    Compute waveforms at time t.

    Args:
        nt (int): Number of time steps.
        dt (float): Time interval.
        fmax (float): Maximum freuqency of Gaussian wave

    Returns:
        waveform_t (Tensor, shape=(nt,)): Waveforms.
    """

    t = (np.arange(0, nt) + 0.5) * dt
    waveform = Gaussian(fmax)
    waveform_t = waveform(t)
    return waveform_t, t


def get_microstrip_filter(air_buffers, npml):
    """ microstrip filter """
    cell_lengths = (0.4064e-3, 0.4233e-3, 0.265e-3)
    obj_lengths = (50 * cell_lengths[0],
                   46 * cell_lengths[1],
                   3 * cell_lengths[2])
    cell_numbers = (
        2 * npml + 2 * air_buffers[0] + int(obj_lengths[0] / cell_lengths[0]),
        2 * npml + 2 * air_buffers[1] + int(obj_lengths[1] / cell_lengths[1]),
        2 * npml + 2 * air_buffers[2] + int(obj_lengths[2] / cell_lengths[2]),
    )

    grid = GridHelper(cell_numbers, cell_lengths, origin=(
        npml + air_buffers[0],
        npml + air_buffers[1],
        npml + air_buffers[2],
    ))

    # Define antenna
    grid[0:50, 0:46, 0:3] = UniformBrick(epsr=2.2)
    grid[14:20, 0:20, 3] = PECPlate('z')
    grid[30:36, 26:46, 3] = PECPlate('z')
    grid[0:50, 20:26, 3] = PECPlate('z')
    grid[0:50, 0:46, 0] = PECPlate('z')

    # Define sources
    grid[14:20, 0, 0:3] = VoltageSource(1., 50., 'zp')

    # Define load
    grid[30:36, 46, 0:3] = Resistor(50., 'z')

    # Define monitors
    grid[14:20, 10, 0:3] = VoltageMonitor('zp')
    grid[14:20, 10, 3] = CurrentMonitor('yp')
    grid[30:36, 36, 0:3] = VoltageMonitor('zp')
    grid[30:36, 36, 3] = CurrentMonitor('yn')

    return grid


def solve(args):
    """solve process"""
    # set up problem
    nt = args.nt
    fmax = args.fmax
    cfl_number = args.cfl_number
    air_buffers = (3, 3, 3)
    npml = 8

    grid_helper = get_microstrip_filter(air_buffers, npml)
    antenna = Antenna(grid_helper)
    ns = len(grid_helper.sources_on_edges)

    cpml = CFSParameters(npml=npml, alpha_max=0.05, sigma_factor=1.3, kappa_max=7, order=3)

    # compute waveforms
    dt = estimate_time_interval(grid_helper.cell_lengths, cfl_number, epsr_min=1., mur_min=1.)

    waveform_t, t = get_waveform_t(nt, dt, fmax)

    # sampling frequencies
    fs = np.linspace(0., fmax, 1001, endpoint=True)

    # define fdtd network
    fdtd_net = full3d.ADFDTD(grid_helper.cell_numbers, grid_helper.cell_lengths, nt, dt, ns, antenna, cpml)

    # define solver
    solver = SParameterSolver(fdtd_net)

    # solve
    _ = solver.solve(waveform_t)

    # eval
    s_parameters = solver.eval(fs, t)

    # show results
    os.makedirs(args.result_dir)
    s_parameters = s_parameters.asnumpy()
    s_complex = s_parameters[..., 0] + 1j * s_parameters[..., 1]
    s11_ref = np.loadtxt(os.path.join(args.dataset_dir, 'microstrip_filter_s11_ref.txt'), delimiter=',')
    s11_ref = s11_ref[np.argsort(s11_ref[:, 0])]
    s21_ref = np.loadtxt(os.path.join(args.dataset_dir, 'microstrip_filter_s21_ref.txt'), delimiter=',')
    s21_ref = s21_ref[np.argsort(s21_ref[:, 0])]
    compare_s(fs, s_complex, os.path.join(args.result_dir, 'microstrip_filter_s_parameters.png'), s11_ref, s21_ref)

    # save results
    np.savez(os.path.join(args.result_dir, 'microstrip_filter_s_parameters.npz'), s_parameters=s_complex, frequency=fs)


if __name__ == '__main__':
    args_ = parse_args()
    if args_.device_target is not None:
        context.set_context(device_target=args_.device_target)
    if args_.device_id is not None:
        context.set_context(device_id=args_.device_id)
    context.set_context(mode=context.GRAPH_MODE, max_call_depth=args_.max_call_depth)
    solve(args_)
