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
init
"""
from .utils import estimate_time_interval, estimate_frequency_resolution
from .utils import zeros, ones, tensor, hstack, vstack, elu
from .cfs_pml import CFSParameters
from .waveforms import BaseWaveform, Gaussian, NormDGaussian, CosineGaussian
from .base_topology_designer import BaseTopologyDesigner
from . import transverse_magnetic
from . import transverse_electric
from . import metric
from .solver import EMInverseSolver


__all__ = [
    "estimate_time_interval",
    "estimate_frequency_resolution",
    "zeros", "ones", "tensor", "hstack", "vstack", "elu",
    "CFSParameters",
    "BaseWaveform", "Gaussian", "NormDGaussian", "CosineGaussian",
    "BaseTopologyDesigner",
    "transverse_magnetic",
    "transverse_electric",
    "metric",
    "EMInverseSolver",
]
