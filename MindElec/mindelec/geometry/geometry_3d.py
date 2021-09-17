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
#pylint: disable=W0223
"""3d geometry"""

from __future__ import absolute_import
import numpy as np

from .geometry_base import Geometry
from .geometry_nd import HyperCube


class Cuboid(HyperCube):
    r"""
    Definition of Cuboid object.

    Args:
        name (str): name of the cuboid.
        coord_min (Union[tuple[int, float], list[int, float], numpy.ndarray]): coordinates of the
            bottom left back corner of cuboid.
        coord_max (Union[tuple[int, float], list[int, float], numpy.ndarray]): coordinates of the
            top right front corner of cuboid.
        dtype (numpy.dtype): data type of sampled point data type. Default: numpy.float32.
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Cuboid
        >>> cuboid_mesh = edict({'domain': edict({'random_sampling': False, 'size': [50, 50, 25]}),
        ...                      'BC': edict({'random_sampling': False, 'size': 1000, 'with_normal': True,}),})
        >>> cuboid = Cuboid("cuboid", (-3.0, 1, 0), (1, 2, 1), sampling_config=create_config_from_edict(cuboid_mesh))
        >>> domain = cuboid.sampling(geom_type="domain")
        >>> bc, bc_normal = cuboid.sampling(geom_type="BC")
        >>> print(domain.shape)
        (62500, 3)
    """
    def __init__(self, name, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(Cuboid, self).__init__(name, 3, coord_min, coord_max, dtype=dtype, sampling_config=sampling_config)


class Sphere(Geometry):
    pass
