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
'''md iteration leap frog'''

import mindspore.numpy as np


def md_iteration_leap_frog(atom_numbers, dt, vel, crd, frc, inverse_mass):
    """
    One step of classical leap frog algorithm to solve the finite difference Hamiltonian equations
    of motion for certain system.

    Args:
        atom_numbers (int): the number of atoms N.
        dt (float32): the simulation time step.
        vel (Tensor, float32): [N, 3], the velocity of each atom.
        crd (Tensor, float32): [N, 3], the coordinate of each atom.
        frc (Tensor, float32): [N, 3], the force felt by each atom.

    Returns:
        acc (Tensor, float32): [N, 3], the acceleration of each atom after updating.
        vel (Tensor, float32): [N, 3], the velocity of each atom after updating.
        crd (Tensor, float32): [N, 3], the coordinate of each atom after updating.
        frc (Tensor, float32): [N, 3], the force felt by each atom after updating.

    Supported Platforms:
        ``GPU``
    """
    acc = np.expand_dims(inverse_mass, -1) * frc
    vel = vel + dt * acc
    crd = crd + dt * vel
    frc = np.zeros([atom_numbers, 3])

    return acc, vel, crd, frc
