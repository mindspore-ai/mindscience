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
"""base classes for geometry"""

from __future__ import absolute_import
from abc import abstractmethod

import copy
import numpy as np
from ....utils import check_param_type, check_param_type_value, check_dict_type_value


GEOM_TYPES = ["domain", "BC", "IC", "time"]
DATA_TYPES = (np.int32, np.int64, np.float16, np.float32, np.float64)
SAMPLER_TYPES = ["uniform", "lhs", "halton", "sobol"]


class PartSamplingConfig:
    """
    Definition of partial sampling configuration.

    Args:
        size (Union[int, tuple[int], list[int]]): Number of sampling points.
        random_sampling (bool): Whether randomly sampling points. Default: ``True``.
        sampler (str): Method for random sampling. Default: ``"uniform"``.
        random_merge (bool): Specifies whether randomly merge coordinates of different dimensions. Default: ``True``.
        with_normal (bool): Specifies whether generating the normal vectors of the boundary. Default: ``False``.
        with_sdf (bool): Specifies whether return the sign-distance-function result of the inner domain points.
                         Default: ``False``.

    Examples:
        >>> from mindscience.data import PartSamplingConfig
        >>> partsampling = PartSamplingConfig(100, True, "uniform", True, True)
    """
    def __init__(self, size, random_sampling=True, sampler="uniform",
                 random_merge=True, with_normal=False, with_sdf=False):
        check_param_type(size, "size", data_type=[int, tuple, list], exclude_type=bool)
        if isinstance(size, (tuple, list)):
            for ele in size:
                check_param_type(ele, "element in size", data_type=int, exclude_type=bool)
        check_param_type(random_sampling, "random_sampling", data_type=bool)
        check_param_type_value(sampler, "sampler", SAMPLER_TYPES, data_type=str)
        check_param_type(random_merge, "random_merge", data_type=bool)
        check_param_type(with_normal, "with_normal", data_type=bool)
        check_param_type(with_sdf, "with_sdf", data_type=bool)

        self.random_sampling = random_sampling
        self.sampler = sampler
        self.size = size
        self.random_merge = random_merge
        self.with_normal = with_normal
        self.with_sdf = with_sdf


class SamplingConfig:
    r"""
    Definition of global sampling configuration.

    Args:
        part_sampling_dict (dict): A dictionary that specifies sampling configurations for different parts.
            Supported keys include ``"domain"``, ``"BC"``, ``"IC"`` and ``"time"``. Each value is a
            :class:`mindscience.data.PartSamplingConfig` instance. Any supported key can be omitted, and the
            corresponding sampling configuration will be set to ``None`` by default.

    Examples:
        >>> from mindscience.data import SamplingConfig, PartSamplingConfig
        >>> part_sampling_config_dict = {"domain" : PartSamplingConfig([100, 100], False, True),
        ...                              "BC" : PartSamplingConfig(100, True, "uniform", True, True)}
        >>> sampling_config = SamplingConfig(part_sampling_config_dict)
    """
    def __init__(self, part_sampling_dict):
        check_dict_type_value(part_sampling_dict, "part_sampling_dict", key_type=str, value_type=PartSamplingConfig,
                              key_value=GEOM_TYPES)
        self.domain = part_sampling_dict.get("domain", None)
        self.bc = part_sampling_dict.get("BC", None)
        self.ic = part_sampling_dict.get("IC", None)
        self.time = part_sampling_dict.get("time", None)
        self._check_size()

    def _check_size(self):
        for geom in [self.domain, self.bc, self.ic, self.time]:
            if geom and not geom.random_sampling:
                if isinstance(geom.size, int) and not isinstance(geom.size, bool):
                    geom.size = [geom.size]
                check_param_type(geom.size, "sample size", data_type=[list, tuple])
                geom.size = np.array(geom.size).astype(np.int64)
                if geom == self.bc:
                    geom.size = np.prod(geom.size)


class Geometry:
    r"""
    Definition of Geometry object.

    Args:
        name (str): Name of the geometry.
        dim (int): Number of dimensions.
        coord_min (Union[int, float, tuple, list, numpy.ndarray]):
            Minimal coordinate of the geometry. If the parameter type is tuple or list, each element should be of type
            int or float, and its length must be consistent with the `dim` parameter.
        coord_max (Union[int, float, tuple, list, numpy.ndarray]):
            Maximal coordinate of the geometry. If the parameter type is tuple or list, each element should be of type
            int or float, and its length must be consistent with the `dim` parameter.
        dtype (numpy.dtype): Data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): Sampling configuration. Default: ``None``.

    Examples:
        >>> from mindscience.data import generate_sampling_config, Geometry
        >>> geometry_config = dict({'domain' : dict({'random_sampling' : True, 'size' : 100}),
        ...                          'BC' : dict({'random_sampling' : True, 'size' : 100, 'sampler' : 'uniform',}),
        ...                          'random_merge' : True,})
        >>> sampling_config = generate_sampling_config(geometry_config)
        >>> geom = Geometry("geom", 1, 0.0, 1.0, sampling_config=sampling_config)
        >>> geom.set_name("geom_name")
    """
    def __init__(self, name, dim, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        check_param_type(name, "geometry name", data_type=str)
        self.name = name
        check_param_type(dim, "dim", data_type=int, exclude_type=bool)
        self.dim = dim
        if self.dim <= 0:
            raise ValueError(f"dimension should not be <= 0, but got dim: {self.dim}")
        supported_type = (int, float, np.ndarray, list, tuple)
        check_param_type(coord_min, "coord_min", data_type=supported_type, exclude_type=bool)
        check_param_type(coord_max, "coord_max", data_type=supported_type, exclude_type=bool)
        if isinstance(coord_min, (int, float)):
            coord_min = [coord_min]
        if isinstance(coord_max, (int, float)):
            coord_max = [coord_max]
        self.coord_min, self.coord_max = np.array(coord_min), np.array(coord_max)
        for ele in self.coord_min:
            check_param_type(ele, "element of coord_min", data_type=DATA_TYPES, exclude_type=bool)
        for ele in self.coord_max:
            check_param_type(ele, "element of coord_max", data_type=DATA_TYPES, exclude_type=bool)
        if len(coord_min) != self.dim or len(coord_max) != self.dim:
            raise ValueError(f"length of coordinates array must be equal with dimension, but got dim: {dim}, "
                             f"coord_min: {coord_min} with length {len(coord_min)}, coord_max {coord_max} "
                             f"with length {len(coord_max)}")
        if dtype not in DATA_TYPES:
            raise TypeError(f"Only data type {DATA_TYPES} are supported, but got {dtype}")
        self.dtype = dtype
        check_param_type(sampling_config, "sampling_config", data_type=(type(None), SamplingConfig))
        self.sampling_config = sampling_config
        self.geom_type = type(self).__name__

    def set_name(self, name):
        """
        Set geometry instance name.

        Args:
            name (str): Name of geometry instance.

        Raises:
            TypeError: If `name` is not string.

        Examples:
            >>> from mindscience.data import generate_sampling_config, Geometry
            >>> geom = Geometry("geom", 1, 0.0, 1.0)
            >>> geom.set_name("geom_name")
        """
        check_param_type(name, "geometry name", data_type=str)
        self.name = name

    def set_sampling_config(self, sampling_config: SamplingConfig):
        """
        Set sampling info.

        Args:
            sampling_config (SamplingConfig): Sampling configuration.

        Raises:
            TypeError: If `sampling_config` is not instance of SamplingConfig.

        Examples:
            >>> from mindscience.data import generate_sampling_config, Geometry
            >>> geometry_config = dict({'domain': dict({'random_sampling': True, 'size': 100}),
            ...                          'BC': dict({'random_sampling': True, 'size': 100, 'sampler': 'uniform',}),
            ...                          'random_merge': True,})
            >>> sampling_config = generate_sampling_config(geometry_config)
            >>> geom = Geometry("geom", 1, 0.0, 1.0)
            >>> geom.set_sampling_config(sampling_config)
        """
        check_param_type(sampling_config, "sampling_config", data_type=SamplingConfig)
        self.sampling_config = copy.deepcopy(sampling_config)

    @abstractmethod
    def _inside(self, points, strict=False):
        raise NotImplementedError(f"{self.geom_type}._inside not implemented")

    @abstractmethod
    def _on_boundary(self, points):
        raise NotImplementedError(f"{self.geom_type}._on_boundary not implemented")

    @abstractmethod
    def sampling(self, geom_type="domain"):
        raise NotImplementedError(f"{self.geom_type}.sampling not implemented")

    @abstractmethod
    def _boundary_normal(self, points):
        raise NotImplementedError(f"{self.geom_type}._boundary_normal not implemented")

    def __and__(self, geom):
        return self.intersection(geom)

    def intersection(self, geom, sampling_config=None):
        from .csg import CSGIntersection  # pylint: disable=import-outside-toplevel
        return CSGIntersection(self, geom, sampling_config)

    def __or__(self, geom):
        return self.union(geom)

    def __sub__(self, geom):
        return self.difference(geom)

    def union(self, geom, sampling_config=None):
        from .csg import CSGUnion  # pylint: disable=import-outside-toplevel
        return CSGUnion(self, geom, sampling_config)

    def __xor__(self, geom):
        return self.exclusive_or(geom)

    def difference(self, geom, sampling_config=None):
        from .csg import CSGDifference  # pylint: disable=import-outside-toplevel
        return CSGDifference(self, geom, sampling_config)

    def exclusive_or(self, geom, sampling_config=None):
        from .csg import CSGXOR  # pylint: disable=import-outside-toplevel
        return CSGXOR(self, geom, sampling_config)
