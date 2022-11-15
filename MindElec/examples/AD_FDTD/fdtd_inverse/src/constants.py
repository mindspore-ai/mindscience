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
#pylint: disable=W0613
"""
Physical constants.
"""
from scipy import constants
import numpy as np

c0 = constants.speed_of_light   # light of speed in the vacuum
epsilon0 = constants.epsilon_0  # vacuum permittivity
mu0 = constants.mu_0            # vacuum permeability
eta0 = np.sqrt(mu0 / epsilon0)  # vacuum wave impedance
