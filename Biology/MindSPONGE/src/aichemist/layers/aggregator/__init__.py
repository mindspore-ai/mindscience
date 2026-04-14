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
Aggregator
"""
from . import node
from . import interaction
from .base import Aggregate
from .node import NodeAggregator, TensorSummation, TensorMean, \
    SoftmaxGeneralizedAggregator, PowermeanGeneralizedAggregator
from .interaction import InteractionAggregator, InteractionSummation, InteractionMean, \
    LinearTransformation, MultipleChannelRepresentation
from .aggregation import MeanAggregation, MaxAggregation, SumAggregation
from .aggregation import SoftmaxAggregation, Set2SetAggregation, SortAggregation

__all__ = []
__all__.extend(node.__all__)
__all__.extend(interaction.__all__)
