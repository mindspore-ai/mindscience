# Copyright 2025 Huawei Technologies Co., Ltd
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
PowerFlowNet MindSpore Network Implementations
Aligned with original PyTorch/torch_geometric structure
"""

from .mpn import (
    MPN, SkipMPN, MaskEmbdMPN,
    MultiMPN, MaskEmbdMultiMPN, MaskEmbdMultiMPNNoMP,
    MultiConvNet, MPNSimplenet, WrappedMultiConv
)
from .gcn import GCNNet
from .mlp import MLPNet

from .gnn_ops import MessagePassing, TAGConv, degree
from .cpu_npu_ops import (
    gather_cpu_npu_compatible,
    pow_cpu_npu_compatible,
    where_cpu_npu_compatible,
    degree_cpu_npu_compatible,
    randint_like_cpu_npu_compatible,
)
from .data_utils import Data, InMemoryDataset, Graph, DataLoader, create_data_splits

from .power_flow_data import (
    PowerFlowData,
    PowerFlowDataLoader,
    PowerFlowDataIterator,
    PowerFlowDataV2,
    PowerFlowDataLoaderV2,
    random_bus_type,
    denormalize,
)

__all__ = [
    'MPN', 'SkipMPN', 'MaskEmbdMPN',
    'MultiMPN', 'MaskEmbdMultiMPN', 'MaskEmbdMultiMPNNoMP',
    'MultiConvNet', 'MPNSimplenet', 'WrappedMultiConv',
    'GCNNet', 'MLPNet',
    'MessagePassing',
    'TAGConv',
    'degree',
    'gather_cpu_npu_compatible',
    'pow_cpu_npu_compatible',
    'where_cpu_npu_compatible',
    'degree_cpu_npu_compatible',
    'randint_like_cpu_npu_compatible',
    'Data',
    'InMemoryDataset',
    'Graph',
    'DataLoader',
    'create_data_splits',
    'PowerFlowData',
    'PowerFlowDataLoader',
    'PowerFlowDataIterator',
    'PowerFlowDataV2',
    'PowerFlowDataLoaderV2',
    'random_bus_type',
    'denormalize',
]
