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
solve process
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mindspore import nn, context

from src import BaseTopologyDesigner
from src import Gaussian, CFSParameters, estimate_time_interval
from src import transverse_magnetic, EMInverseSolver
from src import zeros, tensor, vstack, elu


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(
        description='Electromagnetic Inverse Scattering Solver Based on AD-FDTD')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--device_target', type=str, default=None)
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--nt', type=int, default=350,
                        help='Number of time steps.')
    parser.add_argument('--max_call_depth', type=int, default=2000)
    parser.add_argument('--dataset_dir', type=str,
                        default='./dataset', help='dataset directory')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./ckpt', help='checkpoint directory')
    parser.add_argument('--result_dir', type=str,
                        default='./result', help='result directory')
    options = parser.parse_args()
    return options


class InverseDomain(BaseTopologyDesigner):
    """
    InverseDomain with customized mapping and source locations for user-defined problems.
    """

    def generate_object(self, rho):
        """Generate material tensors.

        Args:
            rho (Parameter): Parameters to be optimized in the inversion domain.

        Returns:
            epsr (Tensor, shape=(self.cell_nunbers)): Relative permittivity in the whole domain.
            sige (Tensor, shape=(self.cell_nunbers)): Conductivity in the whole domain.
        """
        # generate background material tensors
        epsr = self.background_epsr * self.grid
        sige = self.background_sige * self.grid

        # ---------------------------------------------
        # Customized Differentiable Mapping
        # ---------------------------------------------
        epsr[30:70, 30:70] = self.background_epsr + elu(rho, alpha=1e-2)

        return epsr, sige

    def update_sources(self, *args):
        """
        Set locations of sources.

        Args:
            *args: arguments

        Returns:
            jz (Tensor, shape=(ns, 1, nx+1, ny+1)): Jz tensor.
        """
        sources, _, waveform, _ = args
        jz = sources[0]
        jz[0, :, 20, 50] = waveform
        jz[1, :, 50, 20] = waveform
        jz[2, :, 80, 50] = waveform
        jz[3, :, 50, 80] = waveform
        return jz

    def get_outputs_at_each_step(self, *args):
        """Compute output each step.

        Args:
            *args: arguments

        Returns:
            rx (Tensor, shape=(ns, nr)): Ez fields at receivers.
        """
        ez, _, _ = args[0]

        rx = [
            ez[:, 0, 25, 25],
            ez[:, 0, 25, 50],
            ez[:, 0, 25, 75],
            ez[:, 0, 50, 25],
            ez[:, 0, 50, 75],
            ez[:, 0, 75, 25],
            ez[:, 0, 75, 50],
            ez[:, 0, 75, 75],
        ]

        return vstack(rx)


def load_labels(nt, dataset_dir):
    """
    Load labels of Ez fields and epsr.

    Args:
        nt (int): Number of time steps.
        dataset_dir (str): Dataset directory.

    Returns:
        field_labels (Tensor, shape=(nt, ns, nr)): Ez at receivers.
        epsr_labels (Tensor, shape=(nx, ny)): Ground truth for epsr.
    """

    field_label_path = os.path.join(dataset_dir, 'ez_labels.npy')
    field_labels = tensor(np.load(field_label_path))[:nt]

    epsr_label_path = os.path.join(dataset_dir, 'epsr_labels.npy')
    epsr_labels = tensor(np.load(epsr_label_path))

    return field_labels, epsr_labels


def get_waveform_t(nt, dt, fmax):
    """
    Compute waveforms at time t.

    Args:
        nt (int): Number of time steps.
        dt (float): Time interval.
        fmax (float): Maximum freuqency of Gaussian wave

    Returns:
        waveform_t (Tensor, shape=(nt, ns, nr)): Waveforms.
    """

    t = (np.arange(0, nt) + 0.5) * dt
    waveform = Gaussian(fmax)
    waveform_t = waveform(t)
    return waveform_t


def solve(args):
    """solve process"""
    # set up problem
    nt = args.nt
    dataset_dir = args.dataset_dir
    fmax = 1.4e9
    ns = 4
    cell_lengths = (1e-2, 1e-2)
    cell_numbers = (100, 100)
    cfl_number = 0.9
    npml = 10
    rho_init = zeros((40, 40))

    dt = estimate_time_interval(
        cell_lengths, cfl_number, epsr_min=1., mur_min=1.)
    inverse_domain = InverseDomain(cell_numbers, cell_lengths)
    cpml = CFSParameters(npml=npml)

    # load labels
    field_labels, epsr_labels = load_labels(nt, dataset_dir)

    # compute waveforms
    waveform_t = get_waveform_t(nt, dt, fmax)

    # define fdtd network
    fdtd_net = transverse_magnetic.ADFDTD(
        cell_numbers, cell_lengths, nt, dt, ns,
        inverse_domain, cpml, rho_init)

    # define solver for inverse problem
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = nn.Adam(fdtd_net.trainable_params(), learning_rate=args.lr)
    solver = EMInverseSolver(fdtd_net, loss_fn, optimizer)

    # solve
    solver.solve(args.epochs, waveform_t, field_labels)

    # eval
    epsr, _ = solver.eval(epsr_labels)

    # results
    os.makedirs(args.result_dir)
    plt.imshow(epsr.asnumpy().T, origin='lower',
               vmin=1., vmax=4., extent=[0., 1., 0., 1.])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('reconstructed epsr')
    plt.colorbar()
    plt.savefig(os.path.join(args.result_dir, 'epsr_reconstructed.png'))
    plt.close()


if __name__ == '__main__':
    args_ = parse_args()
    if args_.device_target is not None:
        context.set_context(device_target=args_.device_target)
    if args_.device_id is not None:
        context.set_context(device_id=args_.device_id)
    context.set_context(mode=context.PYNATIVE_MODE,  # memory problem unsolved in graph mode
                        max_call_depth=args_.max_call_depth)
    solve(args_)
