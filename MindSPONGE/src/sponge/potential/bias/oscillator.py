# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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
Harmonic oscillator module.
"""
import mindspore as ms
from mindspore import Tensor
from .bias import Bias


class OscillatorBias(Bias):
    """ Add a restraint for heavy atoms in a molecule.

    Args:
        old_crd(Tensor):    The origin coordinates of all atoms.

        k(float):           The elasticity coefficient of all atoms, assuming to be the same.

        nonh_mask(Tensor):  A mask to distinguish H atoms and heavy atoms.

        name (str):         Name of the bias potential. Default: 'oscillator'

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 old_crd: Tensor,
                 k: Tensor,
                 nonh_mask: Tensor,
                 name: str = 'oscillator',
                 ):
        super().__init__(name=name)

        self.old_crd = Tensor(old_crd, ms.float32)
        self.k = Tensor(k, ms.float32)
        self.nonh_mask = Tensor(1 - nonh_mask, ms.int32)

    @ms.jit
    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        shift = coordinate - self.old_crd
        energy = 0.5 * self.k * shift ** 2 * self.nonh_mask
        return energy.sum(-1).sum(1)[None, :]
