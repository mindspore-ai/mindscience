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
"""1d geometry"""

from __future__ import absolute_import
import numpy as np

from .geometry_nd import HyperCube


class Interval(HyperCube):
    r"""
    Definition of Interval object.

    Args:
        name (str): name of the interval.
        coord_min (Union[int, float]): left of the interval.
        coord_max (Union[int, float]): right of the interval.
        dtype (numpy.dtype): Data type of sampled point data type. Default: numpy.float32.
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Raises:
        ValueError: If `coord_min` or `coord_max` is neither int nor float .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Interval
        >>> line_config = edict({'domain': edict({'random_sampling': True, 'size': 100, 'sampler': 'uniform'}),
        ...                      'BC': edict({'random_sampling': True, 'size': 10, 'sampler': 'uniform',}),})
        >>> line = Interval("line", -1.0, 1.0, sampling_config=create_config_from_edict(line_config))
        >>> domain = line.sampling(geom_type="domain")
        >>> bc = line.sampling(geom_type="BC")
        >>> print(bc.shape)
        (10, 1)
    """
    def __init__(self, name, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        if not isinstance(coord_min, (int, float)) or not isinstance(coord_max, (int, float)) or \
            isinstance(coord_min, bool) or isinstance(coord_max, bool):
            raise ValueError("coord_min and coord_max should be int or float for class Interval, but got "
                             "coord_min {} with type {}, coord_max {} with type {}".format(
                                 coord_min, type(coord_min), coord_max, type(coord_max)))
        super(Interval, self).__init__(name, 1, [coord_min], [coord_max], dtype=dtype, sampling_config=sampling_config)
