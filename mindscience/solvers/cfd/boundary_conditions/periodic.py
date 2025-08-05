# Copyright 2022 Huawei Technologies Co., Ltd
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
"""periodic boundary condition"""
from mindspore import jit_class

from .base import Boundary


@jit_class
class Periodic(Boundary):
    """Periodic boundary condition"""

    def __init__(self, config):
        super(Periodic, self).__init__(config)

    def fill_values_head(self, pri_var, axis, pad_size):
        val = pri_var.copy()[:, -pad_size:, :, :]
        return val

    def fill_values_tail(self, pri_var, axis, pad_size):
        val = pri_var.copy()[:, :pad_size, :, :]
        return val
