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
"""s parameter solver"""
import numpy as np
from .utils import compute_dft, tensor, vstack


class SParameterSolver:
    """Solver class for simulating the S parameters.

    Args:
        network (nn.Cell): 3D FDTD network.
    """

    def __init__(self, network) -> None:
        self.network = network
        self.outputs = None

    def solve(self, waveform_t):
        """
        Solve forward problem.

        Args:
            waveform_t (Tensor, shape=(nt,)): Time-domain waveform at (n+0.5)*dt

        Returns:
            outputs (Tensor, shape=(nt, ns, nr, 2)):
                Time-Domain V (= outputs[..., 0]) and I (= outputs[..., 1]) at the monitors.
        """
        self.outputs = self.network(waveform_t)
        return self.outputs

    def eval(self, fs, t):
        """
        Evaluate the S parameters in the frequency-domain.

        Args:
            fs (numpy.ndarray (nf,)): Sampling frequency (Hz).
            t (numpy.ndarray (nt,)): Sampling time (s) of imposed currents.
                (t = (n + 0.5) * dt, n=0,...,nt-1)

        Returns:
            s_parameters (Tensor, shape=(nf,nr,ns,2)):
                Real part (= s_parameters[..., 0]) and Imaginary part (= s_parameters[..., 0])
                of S parameters.
        """
        ns, nr = self.outputs.shape[-3:-1]
        if ns > nr:
            raise ValueError('ns is larger than nr')

        v_dft_r, v_dft_i = compute_dft(self.outputs[:, :, :, 0],
                                       tensor(t) + 0.5 * self.network.dt,
                                       tensor(fs), self.network.dt)
        i_dft_r, i_dft_i = compute_dft(self.outputs[:, :, :, 1],
                                       tensor(t), tensor(fs), self.network.dt)

        z_0 = 50.
        b_r = (v_dft_r - z_0 * i_dft_r) / (2. * np.sqrt(np.abs(z_0)))
        b_i = (v_dft_i - z_0 * i_dft_i) / (2. * np.sqrt(np.abs(z_0)))

        # Assumption: the order of monitors coincides with the order of sources.
        a_r, a_i = [], []
        for i in range(ns):
            a_r.append((v_dft_r[:, i, i] + z_0 * i_dft_r[:, i, i]) /
                       (2. * np.sqrt(np.abs(z_0))))
            a_i.append((v_dft_i[:, i, i] + z_0 * i_dft_i[:, i, i]) /
                       (2. * np.sqrt(np.abs(z_0))))
        a_r = vstack(a_r)[..., None]
        a_i = vstack(a_i)[..., None]

        den = a_r ** 2 + a_i ** 2
        s_r = (b_r * a_r + b_i * a_i) / den
        s_i = (b_i * a_r - b_r * a_i) / den

        s_parameters = vstack([s_r, s_i]).transpose(0, -2, -3, -1)
        return s_parameters
