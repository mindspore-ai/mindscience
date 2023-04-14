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
from ..utils.check_func import check_param_type


class Interval(HyperCube):
    r"""
    Definition of Interval object.

    Args:
        name (str): name of the interval.
        coord_min (Union[int, float]): left of the interval.
        coord_max (Union[int, float]): right of the interval.
        dtype (numpy.dtype): Data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``None``.

    Raises:
        ValueError: If `coord_min` or `coord_max` is neither int nor float .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, Interval
        >>> line_config = dict({'domain': dict({'random_sampling': True, 'size': 100, 'sampler': 'uniform'}),
        ...                      'BC': dict({'random_sampling': True, 'size': 10, 'sampler': 'uniform',}),})
        >>> line = Interval("line", -1.0, 1.0, sampling_config=generate_sampling_config(line_config))
        >>> domain = line.sampling(geom_type="domain")
        >>> bc = line.sampling(geom_type="BC")
        >>> print(bc.shape)
        (10, 1)
    """
    def __init__(self, name, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        check_param_type(coord_min, "coord_min", data_type=(int, float), exclude_type=bool)
        check_param_type(coord_max, "coord_max", data_type=(int, float), exclude_type=bool)
        super(Interval, self).__init__(name, 1, [coord_min], [coord_max], dtype=dtype, sampling_config=sampling_config)
