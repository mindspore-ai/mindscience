# Copyright 2024 Huawei Technologies Co., Ltd
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
"""init"""

import math

from ....e3.o3 import Irreps


class StridedLayout:
    """Utility class to represent a strided layout of a tensor whose irreps all have the same multiplicity."""

    irreps: Irreps
    base_irreps: Irreps
    pad_to_multiple: int
    dim: int
    base_dim: int
    mul: int

    def __init__(self, irreps: Irreps, pad_to_multiple: int = 1):
        irreps = Irreps(irreps)
        self.irreps = irreps
        self.base_irreps = Irreps([(1, ir) for _, ir in irreps])
        # pylint: disable=C1801
        self.mul = self.irreps.data[0].mul if len(irreps) > 0 else 0
        if self.irreps.dim != self.base_irreps.dim * self.mul:
            raise ValueError("wrong irreps")
        self.pad_to_multiple = pad_to_multiple
        if self.pad_to_multiple not in (1, 2, 4, 8):
            raise ValueError("wrong pad_to_multiple")
        dividend = self.pad_to_multiple
        if dividend != 0:
            self.base_dim = int(math.ceil(self.base_irreps.dim / dividend) * self.pad_to_multiple)
        else:
            raise ValueError("pad_to_multiple should not be zero")
        self.dim = self.base_dim * self.mul
