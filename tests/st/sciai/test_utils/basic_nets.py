# Copyright 2023 Huawei Technologies Co., Ltd
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
"""utilities for sciai tests"""
from mindspore import nn


class Net1In0Out(nn.Cell):
    """utilities for sciai tests"""

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(2, 1)

    def construct(self, _):
        pass


class Net1In1OutNumber(nn.Cell):
    """utilities for sciai tests"""

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(2, 1)

    def construct(self, _):
        return 1


class Net1In1Out(nn.Cell):
    """utilities for sciai tests"""

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(2, 1)

    def construct(self, x):
        return self.dense1(x).sum()


class Net1In1OutAbs(nn.Cell):
    """utilities for sciai tests"""

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(2, 1)

    def construct(self, x):
        return self.dense1(x).abs().sum()


class Net1In1OutTensor(nn.Cell):
    """utilities for sciai tests"""

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(2, 1)

    def construct(self, x):
        return self.dense1(x)


class Net1In2Out(nn.Cell):
    """utilities for sciai tests"""

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(2, 1)
        self.dense2 = nn.Dense(2, 1)

    def construct(self, x):
        return self.dense1(x).sum(), self.dense2(x).sum()


class Net2In2Out(nn.Cell):
    """utilities for sciai tests"""

    def construct(self, x, y):
        out1 = 2 * x + y
        out2 = x * x + 4 * x * y + 3 * y
        return out1.sum(), out2.sum()


class Net2In3Out(nn.Cell):
    """utilities for sciai tests"""

    def construct(self, x, y):
        out1 = x + y
        out2 = 2 * x + y
        out3 = x * x + 4 * y * y + 3 * y
        return out1.sum(), out2.sum(), out3.sum()


class Net2In3OutTensor(nn.Cell):
    """utilities for sciai tests"""

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Dense(2, 1, weight_init="ones")
        self.dense2 = nn.Dense(2, 1, weight_init="ones")

    def construct(self, x, y):
        out1 = self.dense1(x)
        out2 = self.dense2(y)
        out3 = 2 * self.dense1(x) + 3 * self.dense2(y)
        return out1.sum(), out2.sum(), out3.sum()


class DuplicatedCell(nn.Cell):
    """utilities for sciai tests"""

    def __init__(self, cell):
        super().__init__()
        for name, single_cell in cell.cells_and_names():
            if name == "":
                continue
            self.__setattr__(name, single_cell)
        self.__setattr__(self.cls_name, cell)

    def construct(self, x):
        return x
