# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
Metrics for collective variables
"""

from mindspore.nn import Metric

from ..colvar import Colvar


class CV(Metric):
    """Metric to output collective variables"""
    def __init__(self,
                 colvar: Colvar,
                 indexes: tuple = (2, 3),
                 ):

        super().__init__()
        self._indexes = indexes
        self.colvar = colvar
        self._cv_value = None

    def clear(self):
        self._cv_value = 0

    def update(self, *inputs):
        coordinate = inputs[self._indexes[0]]
        pbc_box = inputs[self._indexes[1]]
        self._cv_value = self.colvar(coordinate, pbc_box)

    def eval(self):
        return self._cv_value
