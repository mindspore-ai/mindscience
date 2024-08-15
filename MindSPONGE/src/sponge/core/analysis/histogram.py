# Copyright 2021-2024 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
Post analysis cell histogram
"""
import os
import mindspore as ms
from mindspore import Parameter, ops, Tensor
from mindspore import numpy as msnp
from ...metrics import MetricCV
from ...function import calc_gaussian as gaussian

# kT: kJ/mol
KT = 8.31446261815324e-3 * 300
PI = 3.1415926535898


class Histogram(MetricCV):
    """ Histogram post analysis for given cv.
    Args:
        colvar(Colvar): The cv object.
        grid_min(float): The minimum value of cv.
        grid_max(float): The maximum value of cv.
        grid_bins(int): The number of grids of cv.
        sigma(float): Bandwidth of kde function.
        weight(float): The weight at each energy level. default: None.
        periodic(bool): Use the periodic cv or not. default: False.
        traj(bool): Calculate histogram to the trajectory or just to one frame. default: False.
        save_dataset(str): Save the cv dataset to a given file name. default: None.
    """
    def __init__(self,
                 colvar,
                 grid_min=0.,
                 grid_max=1.,
                 grid_bins=20,
                 sigma=0.1,
                 weight=None,
                 periodic=False,
                 traj=False,
                 save_dataset=None):
        super(Histogram, self).__init__(colvar)
        self.grid_size = (grid_max - grid_min) / grid_bins
        grid_bins += 1
        self._shape = (2, grid_bins)
        self.z = self.grid_size * msnp.arange(grid_bins) + grid_min
        if isinstance(sigma, Tensor):
            self.sigma = sigma
        else:
            self.sigma = Tensor(sigma, ms.float32)
        if weight is None:
            self.weight = msnp.ones_like(self.z)
        else:
            if not isinstance(weight, ms.Tensor):
                self.weight = ms.Tensor(weight, ms.float32)
            else:
                self.weight = weight
        if isinstance(periodic, float, int, ms.Tensor) or not periodic:
            self.periodic = periodic
        else:
            self.periodic = grid_max - grid_min
        self._value = Parameter(msnp.zeros((1, grid_bins)), name='hist', requires_grad=True)
        self.traj = traj
        self.save_dataset = save_dataset
        if self.save_dataset is not None:
            self.write_count = 0
            if os.path.exists(self.save_dataset):
                os.remove(self.save_dataset)
            with open(self.save_dataset, 'a+') as file:
                file.write('#!\tFIELDS\ttime\tcv\n')

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def ndim(self) -> int:
        return self._value.ndim

    @property
    def dtype(self) -> type:
        return self._value.dtype

    def clear(self):
        if self.traj:
            pass
        else:
            self._value = ops.assign(self._value, msnp.zeros_like(self._value))

    def _convert_data(self, cv):
        """ Convert the cv data to histogram data"""
        r = self.z - cv
        if self.periodic:
            r = msnp.where(r > self.periodic / 2, r - self.periodic / 2, r)
            r = msnp.where(r < -self.periodic / 2, -r + self.periodic / 2, r)
        res = gaussian(r / self.sigma)
        volume = msnp.sqrt(2*PI) * self.sigma
        height = 1. / volume
        res *= height
        res *= self.weight
        return self._value + res

    def update(self,
               coordinate,
               pbc_box=None,
               energy=None,
               force=None,
               potentials=None,
               total_bias=None,
               biases=None,
               ):
        """

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.
            energy (Tensor):        Tensor of shape (B, 1). Data type is float.
                                    Total potential energy of the simulation system. Default: ``None``.
            force (Tensor):         Tensor of shape (B, A, D). Data type is float.
                                    Force on each atoms of the simulation system. Default: ``None``.
            potentials (Tensor):    Tensor of shape (B, U). Data type is float.
                                    Original potential energies from force field. Default: ``None``.
            total_bias (Tensor):    Tensor of shape (B, 1). Data type is float.
                                    Total bias energy for reweighting. Default: ``None``.
            biases (Tensor):        Tensor of shape (B, V). Data type is float.
                                    Original bias potential energies from bias functions. Default: ``None``.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            A:  Number of atoms of the simulation system.
            D:  Dimension of the space of the simulation system. Usually is 3.
            U:  Number of potential energies.
            V:  Number of bias potential energies.
        """
        #pylint: disable=unused-argument
        colvar = self.colvar(coordinate, pbc_box)
        if self.save_dataset is not None:
            with open(self.save_dataset, 'a+') as file:
                self.write_count += 1
                file.write(str(self.write_count)+'\t'+msnp.array_str(colvar).replace('[', '').replace(']', ''))
                file.write('\n')
        res = self._convert_data(colvar)
        self._value = ops.assign(self._value, res)

    def eval(self):
        return msnp.vstack((self._value, self.z))
