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
import mindspore.numpy as mnp
from .common import get_atom_list_tensor


CONSTANT_kB = mnp.array(0.00198716, mnp.float64)


def md_temperature(residue_numbers, atom_numbers, start, end, atom_vel_f, atom_mass):
    """
    Calculate the MD Temperature.

    Calculate the temperature.

    Supported Platforms:
        ``GPU``
    """
    idx = get_atom_list_tensor(atom_numbers).reshape(1, -1)
    # (M, 1)
    start = start.reshape(-1, 1)
    end = end.reshape(-1, 1)
    # (M, N)
    mask = mnp.logical_and(idx >= start, idx < end).astype('int32')

    # (1, N, 1)
    res = atom_vel_f * atom_mass.reshape(1, -1, 1)
    res = mnp.tile(res, (residue_numbers, 1, 1))
    momentum = mnp.sum(mnp.expand_dims(mask, -1) * res, 1)
    res_mass = mnp.sum(mask * atom_mass.reshape(1, -1), -1)
    ek = 2. * mnp.sum(momentum * momentum, -1) / res_mass * 0.5 / 3. / CONSTANT_kB / residue_numbers
    return ek.astype(mnp.float32)
