# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
Layer module initialization
"""

from .activation import ShiftedSoftplus, Swish
from .norms import MutualInformation, InstanceNorm, PairNorm, \
    LayerNorm, GraphNorm, CoordsNorm
from .conv import MessagePassingBase, GraphConv, GraphAttentionConv, \
    RelationalGraphConv, NeuralFingerprintConv, ChebyshevConv
from .distribution import IndependentGaussian
from .flow import ConditionalFlow
from .pool import DiffPool, MinCutPool
from .sampler import NodeSampler, EdgeSampler
from .common import MLP

from . import aggregator
from . import cutoff
from . import decoder
from . import embedding
from . import interaction
from . import rbf
from . import readout
