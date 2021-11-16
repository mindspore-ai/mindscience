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


constant_kb = mnp.array(0.00198716, mnp.float64)


def md_temperature(mask, atom_vel_f, atom_mass):
    """
    Calculate the MD Temperature.

    Calculate the temperature.

    Supported Platforms:
        ``GPU``
    """
    residue_numbers = mask.shape[0]
    # (1, N, 1)
    res = atom_vel_f * atom_mass.reshape(1, -1, 1)
    res = mnp.tile(res, (residue_numbers, 1, 1))
    momentum = mnp.sum(mnp.expand_dims(mask, -1) * res, 1)
    res_mass = mnp.sum(mask * atom_mass.reshape(1, -1), -1)
    ek = 2. * mnp.sum(momentum * momentum, -1) / res_mass * 0.5 / 3. / constant_kb / residue_numbers
    return ek.astype(mnp.float32)
