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
"""base classes for geometry"""

from __future__ import absolute_import
from abc import abstractmethod
import copy
import numpy as np

GEOM_TYPES = ["domain", "BC", "IC", "time"]
DATA_TYPES = (np.int32, np.int64, np.float16, np.float32, np.float64)


class PartSamplingConfig:
    r"""
    Definition of partial sampling configuration.

    Args:
        size (Union[tuple, list]): number of sampling points.
        random_sampling (bool): Specifies whether randomly sampling points. Default: True.
        sampler (str): method for random sampling. Default: uniform.
        random_merge (bool): Specifies whether randomly merge coordinates of different dimensions. Default: True.
        with_normal (bool): Specifies whether generating the normal vectors of the boundary. Default: False.

    Raises:
        TypeError: size is not int number when random sampling.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindelec.geometry import PartSamplingConfig
        >>> partsampling = PartSamplingConfig(100, True, "uniform", True, True)
    """
    def __init__(self, size, random_sampling=True, sampler="uniform", random_merge=True, with_normal=False):
        if random_sampling and (not isinstance(size, int) or isinstance(size, bool)):
            raise TypeError("The sample size: {} should be int number when random sampling, but got: {}"
                            .format(size, type(size)))
        self.random_sampling = random_sampling
        self.sampler = sampler
        self.size = size
        self.random_merge = random_merge
        self.with_normal = with_normal


class SamplingConfig:
    r"""
    Definition of global sampling configuration.

    Args:
        part_sampling_dict (dict): sampling configuration.

    Raises:
        ValueError: If `coord_min` or `coord_max` is neither int nor float .
        TypeError: If `part_sampling_dict` is not dict.
        KeyError: If `geom_type` not "domain", "BC", "IC" or "time".
        TypeError: If 'config' is not PartSamplingConfig object.
        ValueError: If `self.domain.size` is neither list nor tuple.
        ValueError: If `self.ic.size` is neither list nor tuple.
        ValueError: If `self.time.size` is neither list nor tuple.


    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindelec.geometry import SamplingConfig, PartSamplingConfig
        >>> part_sampling_config_dict = {"domain" : PartSamplingConfig([100, 100], False, True),
        ...                              "BC" : PartSamplingConfig(100, True, "uniform", True, True)}
        >>> sampling_config = SamplingConfig(part_sampling_config_dict)
    """
    def __init__(self, part_sampling_dict):
        _check_dict(part_sampling_dict)
        _check_geom_type(part_sampling_dict)
        self.domain = None if "domain" not in part_sampling_dict.keys() else part_sampling_dict["domain"]
        _check_size_type(self.domain)
        self.bc = None if "BC" not in part_sampling_dict.keys() else part_sampling_dict["BC"]
        if self.bc is not None and not self.bc.random_sampling:
            if isinstance(self.bc.size, (list, tuple)):
                size = np.array(self.bc.size).astype(np.int64)
                self.bc.size = np.prod(size)
            self.bc.size = np.array(self.bc.size).astype(np.int64)

        self.ic = None if "IC" not in part_sampling_dict.keys() else part_sampling_dict["IC"]
        _check_size_type(self.ic)

        self.time = None if "time" not in part_sampling_dict.keys() else part_sampling_dict["time"]
        _check_size_type(self.time)


class Geometry:
    r"""
    Definition of Geometry object.

    Args:
        name (str): name of the geometry.
        dim (int): number of dimensions.
        coord_min (Union[int, float, list[int, float], tuple[int, float], numpy.ndarray]): minimal coordinate of
            the geometry.
        coord_max (Union[int, float, list[int, float], tuple[int, float], numpy.ndarray]): maximal coordinate of
            the geometry.
        dtype (numpy.dtype): Data type of sampled point data type. Default: numpy.float32.
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Geometry
        >>> geometry_config = edict({'domain' : edict({'random_sampling' : True, 'size' : 100}),
        ...                          'BC' : edict({'random_sampling' : True, 'size' : 100, 'sampler' : 'uniform',}),
        ...                          'random_merge' : True,})
        >>> sampling_config = create_config_from_edict(geometry_config)
        >>> geom = Geometry("geom", 1, 0.0, 1.0, sampling_config=sampling_config)
        >>> geom.set_name("geom_name")
    """
    def __init__(self, name, dim, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        if not isinstance(name, str):
            raise TypeError("geometry name must be string, but got: {}, type: {}".format(name, type(name)))
        self.name = name
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("dim type should be integer, but got dim: {}, type: {}".format(dim, type(dim)))
        self.dim = dim
        if self.dim <= 0:
            raise ValueError("dimension should not be non-positive, but got dim: {}".format(self.dim))
        if isinstance(coord_min, (int, float)):
            coord_min = [coord_min]
        if isinstance(coord_max, (int, float)):
            coord_max = [coord_max]
        if not isinstance(coord_min, (np.ndarray, list, tuple)) or not isinstance(coord_max, (np.ndarray, list, tuple)):
            raise TypeError("argument coord_min/max must be np.array or list or tuple, but got coord_min: {}, type: {};"
                            "coord_max: {}, type: {}".format(coord_min, type(coord_min), coord_max, type(coord_max)))
        self.coord_min, self.coord_max = np.array(coord_min), np.array(coord_max)
        for ele in np.concatenate((self.coord_min, self.coord_max), axis=0):
            if not isinstance(ele, DATA_TYPES):
                raise TypeError("element for coord should be int or float, but got ele: {}, type: {}".format(
                    ele, type(ele)))
        if len(coord_min) != self.dim or len(coord_max) != self.dim:
            raise ValueError("dimension of coordinates array must be equal with dim, but got dim: {},"
                             "coord_min: {} with length {}, coord_max {} with length {}".format(
                                 dim, coord_min, len(coord_min), coord_max, len(coord_max)))
        if np.any(self.coord_max - self.coord_min <= 0.0):
            raise ValueError("coord_min should be smaller than coord_max, but got coord_min: {}, coor_max: {}".format(
                self.coord_min, self.coord_max))

        if dtype not in DATA_TYPES:
            raise TypeError("Unsupported datatype: {}, only: {} are supported".format(dtype, DATA_TYPES))
        self.dtype = dtype
        if sampling_config is not None and not isinstance(sampling_config, SamplingConfig):
            raise TypeError("sampling_config should be instance of SamplingConfig, bug got {} with type {}".format(
                sampling_config, type(sampling_config)
            ))
        self.sampling_config = sampling_config
        self.geom_type = type(self).__name__

    def set_name(self, name):
        """
        set geometry instance name

        Args:
            name (str): name of geometry instance

        Raises:
            TypeError: If `name` is not string.

        Examples:
            >>> from mindelec.geometry import create_config_from_edict, Geometry
            >>> geom = Geometry("geom", 1, 0.0, 1.0)
            >>> geom.set_name("geom_name")
        """
        if not isinstance(name, str):
            raise TypeError("geometry name must be string, but got: {}, type: {}".format(name, type(name)))
        self.name = name

    def set_sampling_config(self, sampling_config: SamplingConfig):
        """
        set sampling info

        Args:
            sampling_config (SamplingConfig): sampling configuration.

        Raises:
            TypeError: If `sampling_config` is not instance of SamplingConfig.

        Examples:
            >>> from easydict import EasyDict as edict
            >>> from mindelec.geometry import create_config_from_edict, Geometry
            >>> geometry_config = edict({'domain': edict({'random_sampling': True, 'size': 100}),
            ...                          'BC': edict({'random_sampling': True, 'size': 100, 'sampler': 'uniform',}),
            ...                          'random_merge': True,})
            >>> sampling_config = create_config_from_edict(geometry_config)
            >>> geom = Geometry("geom", 1, 0.0, 1.0)
            >>> geom.set_sampling_config(sampling_config)
        """
        if not isinstance(sampling_config, SamplingConfig):
            raise TypeError("sampling_config: {} should be instance of class SamplingConfig, bug got: {}".format(
                sampling_config, type(sampling_config)))
        self.sampling_config = copy.deepcopy(sampling_config)

    @abstractmethod
    def _inside(self, points, strict=False):
        raise NotImplementedError("{}._inside not implemented".format(self.geom_type))

    @abstractmethod
    def _on_boundary(self, points):
        raise NotImplementedError("{}._on_boundary not implemented".format(self.geom_type))

    @abstractmethod
    def _boundary_normal(self, points):
        raise NotImplementedError("{}._boundary_normal not implemented".format(self.geom_type))

    @abstractmethod
    def sampling(self, geom_type="domain"):
        raise NotImplementedError("{}.sampling not implemented".format(self.geom_type))

    def __and__(self, geom):
        return self.intersection(geom)

    def intersection(self, geom, sampling_config=None):
        from .csg import CSGIntersection
        return CSGIntersection(self, geom, sampling_config)

    def __or__(self, geom):
        return self.union(geom)

    def union(self, geom, sampling_config=None):
        from .csg import CSGUnion

        return CSGUnion(self, geom, sampling_config)

    def __sub__(self, geom):
        return self.difference(geom)

    def difference(self, geom, sampling_config=None):
        from .csg import CSGDifference
        return CSGDifference(self, geom, sampling_config)

    def __xor__(self, geom):
        return self.exclusive_or(geom)

    def exclusive_or(self, geom, sampling_config=None):
        from .csg import CSGXOR
        return CSGXOR(self, geom, sampling_config)


def _check_dict(part_sampling_dict):
    """check whether the type is dict"""
    if not isinstance(part_sampling_dict, dict):
        raise TypeError("part_sampling_dict: {} should be type of dict, the key is geom_type while the value is "
                        "instance of class PartSamplinlgConfiwhich describe the sampling info of each part"
                        .format(part_sampling_dict))


def _check_geom_type(part_sampling_dict):
    """check key and value of part_sampling_dict"""
    for geom_type in part_sampling_dict.keys():
        if geom_type not in GEOM_TYPES:
            raise KeyError("Unknown geom types: {}, only {} are supported now"
                           .format(geom_type, GEOM_TYPES))
        config = part_sampling_dict[geom_type]
        if not isinstance(config, PartSamplingConfig):
            raise TypeError("Wrong type of value: {}, should be inance of class PartSamplingConfig, but got: {}"
                            .format(config, type(config)))


def _check_size_type(geom):
    """check size type of specified geometry"""
    if geom is not None and not geom.random_sampling:
        if isinstance(geom.size, int) and not isinstance(geom.size, bool):
            geom.size = [geom.size]
        if not isinstance(geom.size, (list, tuple)):
            raise ValueError("The sample size: {} should be type of list/tuple which length equal to the geom's "
                             "dimension when sampling on regular mesh".format(geom.size))
        geom.size = np.array(geom.size).astype(np.int64)
