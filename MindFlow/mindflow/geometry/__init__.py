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
"""init"""
from .geometry_base import Geometry, PartSamplingConfig, SamplingConfig
from .geometry_1d import Interval
from .geometry_2d import Disk, Rectangle, Triangle, Pentagon
from .geometry_3d import Cuboid, Cylinder, Cone, Tetrahedron
from .geometry_nd import FixedPoint, HyperCube
from .geometry_td import TimeDomain, GeometryWithTime
from .csg import CSGIntersection, CSGDifference, CSGUnion, CSGXOR, CSG
from .geom_utils import generate_sampling_config

__all__ = [
    "Geometry",
    "PartSamplingConfig",
    "SamplingConfig",
    "Interval",
    "Disk",
    "Rectangle",
    "Triangle",
    "Pentagon",
    "Cuboid",
    "Cylinder",
    "Cone",
    "Tetrahedron",
    "FixedPoint",
    "HyperCube",
    "TimeDomain",
    "GeometryWithTime",
    "CSGIntersection",
    "CSGDifference",
    "CSGUnion",
    "CSGXOR",
    "generate_sampling_config"
]

__all__.sort()
