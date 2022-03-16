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
'''md temperature'''
import mindspore.numpy as np
from mindspore.ops.operations import _csr_ops


constant_kb = np.array(0.00198716, np.float64)
csr_reducesum = _csr_ops.CSRReduceSum()

def md_temperature(mask, atom_vel_f, atom_mass):
    """
    Calculate the MD Temperature.

    Calculate the temperature.

    Supported Platforms:
        ``GPU``
    """
    residue_numbers = mask.shape[0]
    # (n, 3) * (n, 1) -> (n, 3)
    res = atom_vel_f * atom_mass.reshape(-1, 1)
    # (n, 1)
    res_x, res_y, res_z = np.split(res, 3, axis=1)
    # sparse(m, n) * dense(1, n) -> sparse(m, n)
    momentum_x = mask * res_x.reshape(1, -1)
    momentum_y = mask * res_y.reshape(1, -1)
    momentum_z = mask * res_z.reshape(1, -1)
    # sparse(m, n) -> dense(m, 1)
    momentum_x = csr_reducesum(momentum_x, 1)
    momentum_y = csr_reducesum(momentum_y, 1)
    momentum_z = csr_reducesum(momentum_z, 1)
    # dense(m, 1) -> dense(m, 1)
    momentum = momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z

    # sparse(m, n) * dense(1, n) -> sparse(m, n)
    res_mass = mask * atom_mass.reshape(1, -1)
    # sparse(m, n) -> dense(m, 1)
    res_mass = csr_reducesum(res_mass, 1)
    n = 3. * residue_numbers
    ek = momentum / res_mass / n / constant_kb

    return ek.astype(np.float32)
