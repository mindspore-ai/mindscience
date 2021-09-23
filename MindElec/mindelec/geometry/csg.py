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
#pylint: disable=W0221
#pylint: disable=W0212
"""Constructive Solid Geometry"""
from __future__ import absolute_import

import copy
import numpy as np

from mindspore import log as logger

from .geometry_base import Geometry, SamplingConfig, GEOM_TYPES
from .utils import sample

def _check_geom(geoms):
    for i in range(len(geoms)):
        if not isinstance(geoms[i], Geometry):
            raise TypeError("geom{} should be instance of class Geometry, but got: {}".format(
                i + 1, type(geoms[i])))
    if geoms[0].dim != geoms[1].dim:
        raise ValueError("Mismatch of dimension, geom1: {}'s dim is: {} while geom: {}'s dim is: {}.".format(
            geoms[0].name, geoms[0].dim, geoms[1].name, geoms[1].dim))


class CSG(Geometry):
    r"""
    CSG base class.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, name, geom1, geom2, coord_min, coord_max, sampling_config=None):
        _check_geom([geom1, geom2])
        self.geom_type = type(self).__name__
        geom1 = copy.deepcopy(geom1)
        geom2 = copy.deepcopy(geom2)
        self.geom1 = geom1
        self.geom2 = geom2

        super(CSG, self).__init__(name, geom1.dim, coord_min, coord_max, geom1.dtype, sampling_config)

    def _check_sampling_config(self, sampling_config):
        """check sampling_config"""
        if sampling_config is None:
            raise ValueError("Sampling config for {}:{} should not be None, please call set_sampling_config to reset "
                             "this info".format(self.geom_type, self.name))
        if not isinstance(sampling_config, SamplingConfig):
            raise TypeError("sampling_config should be instance of SamplingConfig, but got {} with type {}".format(
                sampling_config, type(sampling_config)
            ))
        if sampling_config.domain is not None:
            if not sampling_config.domain.random_sampling:
                raise ValueError("Only random sampling strategy is supported for CSG instance in domain")
        if sampling_config.bc is not None:
            if not sampling_config.bc.random_sampling:
                raise ValueError("Only random sampling strategy is supported for CSG instance in bc")
        if sampling_config.ic is not None:
            if not sampling_config.ic.random_sampling:
                raise ValueError("Only random sampling strategy is supported for CSG instance in ic")

    def _random_domain_points(self):
        raise NotImplementedError("{}._random_domain_points not implemented".format(self.geom_type))

    def _random_boundary_points(self):
        raise NotImplementedError("{}._random_boundary_points not implemented".format(self.geom_type))

    def set_sampling_config(self, sampling_config: SamplingConfig):
        """
        set sampling info

        Args:
            sampling_config (SamplingConfig): sampling configuration.

        Raises:
            TypeError: If `sampling_config` is not instance of SamplingConfig.
        """
        self._check_sampling_config(sampling_config)
        self.sampling_config = copy.deepcopy(sampling_config)
        self.geom1.set_sampling_config(self.sampling_config)
        self.geom2.set_sampling_config(self.sampling_config)

    def set_name(self, name):
        """
        set geometry instance name
        """
        if not isinstance(name, str):
            raise TypeError("geometry name must be string, but got: {}, type: {}".format(name, type(name)))
        self.name = name

    def sampling(self, geom_type="domain"):
        """
        sampling points

        Args:
            geom_type (str): geometry type

        Returns:
            Numpy.array, numpy array with or without boundary normal vectors

        Raises:
            ValueError: If `config` is None.
            KeyError: If `geom_type` is `domain` but `config.domain` is None.
            KeyError: If `geom_type` is `BC` but `config.bc` is None.
            ValueError: If `geom_type` is neither `BC` nor `domain`.
        """
        self._check_sampling_config(self.sampling_config)
        config = self.sampling_config
        if not isinstance(geom_type, str):
            raise TypeError("geom type shouild be string, but got {} with type {}".format(geom_type, type(geom_type)))
        if geom_type not in GEOM_TYPES:
            raise ValueError("Unknown geom type: {}, only {} are supported now".format(geom_type, GEOM_TYPES))
        if geom_type.lower() == "domain":
            if config.domain is None:
                raise KeyError("Sampling config for domain of {}:{} should not be none"
                               .format(self.geom_type, self.name))
            logger.info("Sampling domain points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))
            column_name = self.name + "_domain_points"
            data = self._random_domain_points()
            self.columns_dict["domain"] = [column_name]
            data = data.astype(self.dtype)
            return data
        if geom_type.lower() == "bc":
            if config.bc is None:
                raise KeyError("Sampling config for BC of {}:{} should not be none".format(self.geom_type, self.name))
            logger.info("Sampling BC points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))
            if config.bc.with_normal:
                data, data_normal = self._random_boundary_points()
                column_data = self.name + "_BC_points"
                column_normal = self.name + "_BC_normal"
                self.columns_dict["BC"] = [column_data, column_normal]
                data = data.astype(self.dtype)
                data_normal = data_normal.astype(self.dtype)
                return data, data_normal
            data = self._random_boundary_points()
            column_data = self.name + "_BC_points"
            self.columns_dict["BC"] = [column_data]
            data = data.astype(self.dtype)
            return data
        raise ValueError("Unknown geom_type: {}, only \"domain/BC\" are supported for {}:{}".format(
            geom_type, self.geom_type, self.name))


class CSGDifference(CSG):
    r"""
    CSG class for difference of geometry.

    Args:
        geom1 (Geometry): a geometry object
        geom2 (Geometry): a geometry object to be subtracted from geom1
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Disk, Rectangle, CSGDifference
        >>> sampling_config_csg = edict({
        ...     'domain': edict({
        ...         'random_sampling': True,
        ...         'size': 1000,
        ...         'sampler': 'uniform'
        ...     }),
        ...     'BC': edict({
        ...         'random_sampling': True,
        ...         'size': 200,
        ...         'sampler': 'uniform',
        ...         'with_normal': True,
        ...     }),
        ... })
        >>> disk = Disk("disk", (1.2, 0.5), 0.8)
        >>> rect = Rectangle("rect", (-1.0, 0), (1, 1))
        >>> diff = CSGDifference(rect, disk)
        >>> diff.set_sampling_config(create_config_from_edict(sampling_config_csg))
        >>> domain = diff.sampling(geom_type="domain")
        >>> bc, bc_normal = diff.sampling(geom_type="bc")
        >>> print(domain.shape)
        (1000, 2)
    """
    def __init__(self, geom1, geom2, sampling_config=None):
        r"""This class returns geom1\\geom2"""
        _check_geom([geom1, geom2])
        name = geom1.name + "_sub_" + geom2.name
        self.columns_dict = {}
        super(CSGDifference, self).__init__(name, geom1, geom2, geom1.coord_min, geom1.coord_max, sampling_config)
        if sampling_config is None:
            sampling_config = geom1.sampling_config
        else:
            self.set_sampling_config(sampling_config)

    def _inside(self, points):
        """Check whether points in geom1\\geom2 or not"""
        inside1 = self.geom1._inside(points)
        inside2 = self.geom2._inside(points)
        return np.logical_and(inside1, ~inside2)

    def _random_domain_points(self):
        """Sample points in geom1\\geom2"""
        diff_size = self.sampling_config.domain.size
        diff_domain_points = np.empty(shape=(diff_size, self.dim))
        index = 0
        while index < diff_size:
            domain_points_from_geom1 = self.geom1.sampling(geom_type="domain")
            domain_points_from_geom1_sub_geom2 = domain_points_from_geom1[~self.geom2._inside(domain_points_from_geom1)]
            added_size = len(domain_points_from_geom1_sub_geom2)

            diff_domain_points[index: min(diff_size, added_size + index)] = \
                domain_points_from_geom1_sub_geom2[:min(added_size, diff_size - index)]
            index += added_size

        return diff_domain_points

    def _random_boundary_points(self):
        """Sample boundary points in geom1\\geom2"""
        diff_size = self.sampling_config.bc.size
        need_normal = self.sampling_config.bc.with_normal
        diff_points = np.empty(shape=(diff_size, self.dim))
        if need_normal:
            diff_normal = np.empty(shape=(diff_size, self.dim))

        index = 0
        while index < diff_size:
            if need_normal:
                points_from_geom1, normal_from_geom1 = self.geom1.sampling(geom_type="BC")
            else:
                points_from_geom1 = self.geom1.sampling(geom_type="BC")

            points_from_geom1_out_geom2 = points_from_geom1[~self.geom2._inside(points_from_geom1)]
            if need_normal:
                normal_from_geom1_out_geom2 = normal_from_geom1[~self.geom2._inside(points_from_geom1)]

            if need_normal:
                points_from_geom2, normal_from_geom2 = self.geom2.sampling(geom_type="BC")
            else:
                points_from_geom2 = self.geom2.sampling(geom_type="BC")

            points_from_geom2_out_geom1 = points_from_geom2[self.geom1._inside(points_from_geom2)]
            if need_normal:
                normal_from_geom2_out_geom1 = -1 * normal_from_geom2[self.geom1._inside(points_from_geom2)]

            points_from_geom1_sub_geom2 = np.concatenate([points_from_geom1_out_geom2, points_from_geom2_out_geom1],
                                                         axis=0)

            added_size = len(points_from_geom1_sub_geom2)
            if need_normal:
                rand_index = np.random.permutation(added_size)
                points_from_geom1_sub_geom2 = points_from_geom1_sub_geom2[rand_index]
                normal_from_geom1_sub_geom2 = np.concatenate([normal_from_geom1_out_geom2,
                                                              normal_from_geom2_out_geom1], axis=0)
                normal_from_geom1_sub_geom2 = normal_from_geom1_sub_geom2[rand_index]
            else:
                points_from_geom1_sub_geom2 = np.random.permutation(points_from_geom1_sub_geom2)

            diff_points[index: min(diff_size, added_size + index)] = \
                points_from_geom1_sub_geom2[:min(added_size, diff_size - index)]
            if need_normal:
                diff_normal[index: min(diff_size, added_size + index)] = \
                    normal_from_geom1_sub_geom2[:min(added_size, diff_size - index)]
            index += added_size

        if need_normal:
            return diff_points, diff_normal
        return diff_points


class CSGUnion(CSG):
    r"""
    CSG class for union of geometries.

    Args:
        geom1 (Geometry): a geometry object
        geom2 (Geometry): a geometry object to be subtracted from geom1
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Disk, Rectangle, CSGUnion
        >>> sampling_config_csg = edict({
        ...     'domain': edict({
        ...         'random_sampling': True,
        ...         'size': 1000,
        ...         'sampler': 'uniform'
        ...     }),
        ...     'BC': edict({
        ...         'random_sampling': True,
        ...         'size': 200,
        ...         'sampler': 'uniform',
        ...         'with_normal': True,
        ...     }),
        ... })
        >>> disk = Disk("disk", (1.2, 0.5), 0.8)
        >>> rect = Rectangle("rect", (-1.0, 0), (1, 1))
        >>> union = CSGUnion(rect, disk)
        >>> union.set_sampling_config(create_config_from_edict(sampling_config_csg))
        >>> domain = union.sampling(geom_type="domain")
        >>> bc, bc_normal = union.sampling(geom_type="bc")
        >>> print(domain.shape)
        (1000, 2)
    """

    def __init__(self, geom1, geom2, sampling_config=None):
        """This class returns geom1 or geom2"""
        _check_geom([geom1, geom2])
        name = geom1.name + "_add_" + geom2.name
        self.columns_dict = {}
        min_coord_min = np.minimum(geom1.coord_min, geom2.coord_min)
        max_coord_max = np.maximum(geom1.coord_max, geom2.coord_max)
        super(CSGUnion, self).__init__(name, geom1, geom2, min_coord_min, max_coord_max, sampling_config)
        if sampling_config is None:
            self.sampling_config = None
        else:
            self.set_sampling_config(sampling_config)

    def _inside(self, points):
        """Check whether points in geom1 or geom2 or not"""
        inside1 = self.geom1._inside(points)
        inside2 = self.geom2._inside(points)
        return np.logical_or(inside1, inside2)

    def _random_domain_points(self):
        """Sample points in geom1 or geom2"""
        union_size = self.sampling_config.domain.size
        sampler = self.sampling_config.domain.sampler
        union_domain_points = np.empty(shape=(union_size, self.dim))

        index = 0
        while index < union_size:
            domain_points = sample(union_size, self.dim, sampler) * (self.coord_max - self.coord_min) + self.coord_min
            domain_points = np.reshape(domain_points, (-1, self.dim))
            union_points = domain_points[self._inside(domain_points)]
            added_size = len(union_points)
            union_domain_points[index: min(union_size, added_size + index)] = \
                union_points[:min(added_size, union_size - index)]
            index += added_size
        return union_domain_points

    def _random_boundary_points(self):
        """Sample boundary points in geom1 or geom2"""
        union_size = self.sampling_config.bc.size
        need_normal = self.sampling_config.bc.with_normal
        union_points = np.empty(shape=(union_size, self.dim))
        if need_normal:
            union_normal = np.empty(shape=(union_size, self.dim))

        index = 0
        if need_normal:
            while index < union_size:
                boundary_from_geom1, normal_from_geom1 = self.geom1.sampling(geom_type="BC")
                boundary_from_geom2, normal_from_geom2 = self.geom2.sampling(geom_type="BC")
                bound_geom1_sub_geom2 = boundary_from_geom1[~self.geom2._inside(boundary_from_geom1)]
                normal_geom1_sub_geom2 = normal_from_geom1[~self.geom2._inside(boundary_from_geom1)]
                bound_geom2_sub_geom1 = boundary_from_geom2[~self.geom1._inside(boundary_from_geom2)]
                normal_geom2_sub_geom1 = normal_from_geom2[~self.geom1._inside(boundary_from_geom2)]
                boundary_from_csg = np.concatenate((bound_geom1_sub_geom2, bound_geom2_sub_geom1), axis=0)
                normal_from_csg = np.concatenate((normal_geom1_sub_geom2, normal_geom2_sub_geom1), axis=0)
                added_size = len(boundary_from_csg)
                rand_index = np.random.permutation(added_size)
                boundary_from_csg = boundary_from_csg[rand_index]
                normal_from_csg = normal_from_csg[rand_index]
                union_points[index: min(union_size, added_size + index)] = \
                    boundary_from_csg[:min(added_size, union_size - index)]
                union_normal[index: min(union_size, added_size + index)] = \
                    normal_from_csg[:min(added_size, union_size - index)]
                index += added_size
            return union_points, union_normal

        while index < union_size:
            boundary_from_geom1 = self.geom1.sampling(geom_type="BC")
            boundary_from_geom2 = self.geom2.sampling(geom_type="BC")
            bound_geom1_sub_geom2 = boundary_from_geom1[~self.geom2._inside(boundary_from_geom1)]
            bound_geom2_sub_geom1 = boundary_from_geom2[~self.geom1._inside(boundary_from_geom2)]
            boundary_from_csg = np.concatenate((bound_geom1_sub_geom2, bound_geom2_sub_geom1), axis=0)
            added_size = len(boundary_from_csg)
            boundary_from_csg = np.random.permutation(boundary_from_csg)
            union_points[index: min(union_size, added_size + index)] = \
                boundary_from_csg[:min(added_size, union_size - index)]
            index += added_size
        return union_points


class CSGIntersection(CSG):
    r"""
    CSG class for intersection of geometries.

    Args:
        geom1 (Geometry): a geometry object
        geom2 (Geometry): a geometry object to be subtracted from geom1
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Disk, Rectangle, CSGIntersection
        >>> sampling_config_csg = edict({
        ...     'domain': edict({
        ...         'random_sampling': True,
        ...         'size': 1000,
        ...         'sampler': 'uniform'
        ...     }),
        ...     'BC': edict({
        ...         'random_sampling': True,
        ...         'size': 200,
        ...         'sampler': 'uniform',
        ...         'with_normal': True,
        ...     }),
        ... })
        >>> disk = Disk("disk", (1.2, 0.5), 0.8)
        >>> rect = Rectangle("rect", (-1.0, 0), (1, 1))
        >>> inter = CSGIntersection(rect, disk)
        >>> inter.set_sampling_config(create_config_from_edict(sampling_config_csg))
        >>> domain = inter.sampling(geom_type="domain")
        >>> bc, bc_normal = inter.sampling(geom_type="bc")
        >>> print(domain.shape)
        (1000, 2)
    """

    def __init__(self, geom1, geom2, sampling_config=None):
        """This class returns geom1 and geom2"""
        _check_geom([geom1, geom2])
        if geom1.dim != geom2.dim:
            raise ValueError("Unable to union: {} and {} do not match in dimension.".format(geom1.name, geom2.name))
        name = geom1.name + "_add_" + geom2.name
        self.columns_dict = {}
        max_coord_min = np.maximum(geom1.coord_min, geom2.coord_min)
        min_coord_max = np.minimum(geom1.coord_max, geom2.coord_max)
        super(CSGIntersection, self).__init__(name, geom1, geom2, max_coord_min, min_coord_max, sampling_config)
        if sampling_config is None:
            sampling_config = geom1.sampling_config
        else:
            self.set_sampling_config(sampling_config)

    def _inside(self, points):
        """Check whether points in geom1 and geom2 or not"""
        inside_geom1 = self.geom1._inside(points)
        inside_geom2 = self.geom2._inside(points)
        return np.logical_and(inside_geom1, inside_geom2)

    def _random_domain_points(self):
        """Sample points in geom1 and geom2"""
        inter_size = self.sampling_config.domain.size
        sampler = self.sampling_config.domain.sampler
        inter_domain_points = np.empty(shape=(inter_size, self.dim))

        index = 0
        while index < inter_size:
            domain_points = sample(inter_size, self.dim, sampler) * (self.coord_max - self.coord_min) + self.coord_min
            domain_points = np.reshape(domain_points, (-1, self.dim))
            inter_points = domain_points[self._inside(domain_points)]
            added_size = len(inter_points)
            inter_domain_points[index: min(inter_size, added_size + index)] = \
                inter_points[:min(added_size, inter_size - index)]
            index += added_size
        return inter_domain_points

    def _random_boundary_points(self):
        """Sample points in geom1 and geom2"""
        inter_size = self.sampling_config.bc.size
        need_normal = self.sampling_config.bc.with_normal
        inter_points = np.empty(shape=(inter_size, self.dim))
        if need_normal:
            inter_normal = np.empty(shape=(inter_size, self.dim))

        index = 0
        if need_normal:
            while index < inter_size:
                boundary_from_geom1, normal_from_geom1 = self.geom1.sampling(geom_type="BC")
                boundary_from_geom2, normal_from_geom2 = self.geom2.sampling(geom_type="BC")
                boundary_from_geom1_exclude = boundary_from_geom1[self.geom2._inside(boundary_from_geom1)]
                normal_from_geom1_exclude = normal_from_geom1[self.geom2._inside(boundary_from_geom1)]
                boundary_from_geom2_exclude = boundary_from_geom2[self.geom1._inside(boundary_from_geom2)]
                normal_from_geom2_exclude = normal_from_geom2[self.geom1._inside(boundary_from_geom2)]
                boundary_from_csg = np.concatenate((boundary_from_geom1_exclude, boundary_from_geom2_exclude), axis=0)
                normal_from_csg = np.concatenate((normal_from_geom1_exclude, normal_from_geom2_exclude), axis=0)
                added_size = len(boundary_from_csg)
                rand_index = np.random.permutation(added_size)
                boundary_from_csg = boundary_from_csg[rand_index]
                normal_from_csg = normal_from_csg[rand_index]
                inter_points[index: min(inter_size, added_size + index)] = \
                    boundary_from_csg[:min(added_size, inter_size - index)]
                inter_normal[index: min(inter_size, added_size + index)] = \
                    normal_from_csg[:min(added_size, inter_size - index)]
                index += added_size
            return inter_points, inter_normal

        while index < inter_size:
            boundary_from_geom1 = self.geom1.sampling(geom_type="BC")
            boundary_from_geom2 = self.geom2.sampling(geom_type="BC")
            boundary_from_geom1_exclude = boundary_from_geom1[self.geom2._inside(boundary_from_geom1)]
            boundary_from_geom2_exclude = boundary_from_geom2[self.geom1._inside(boundary_from_geom2)]
            boundary_from_csg = np.concatenate((boundary_from_geom1_exclude, boundary_from_geom2_exclude), axis=0)
            added_size = len(boundary_from_csg)
            boundary_from_csg = np.random.permutation(boundary_from_csg)
            inter_points[index: min(inter_size, added_size + index)] = \
                boundary_from_csg[:min(added_size, inter_size - index)]
            index += added_size
        return inter_points


class CSGXOR(CSG):
    r"""
    CSG class for xor of geometries.

    Args:
        geom1 (Geometry): a geometry object
        geom2 (Geometry): a geometry object to be subtracted from geom1
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Disk, Rectangle, CSGXOR
        >>> sampling_config_csg = edict({
        ...     'domain': edict({
        ...         'random_sampling': True,
        ...         'size': 1000,
        ...         'sampler': 'uniform'
        ...     }),
        ...     'BC': edict({
        ...         'random_sampling': True,
        ...         'size': 200,
        ...         'sampler': 'uniform',
        ...         'with_normal': True,
        ...     }),
        ... })
        >>> disk = Disk("disk", (1.2, 0.5), 0.8)
        >>> rect = Rectangle("rect", (-1.0, 0), (1, 1))
        >>> xor = CSGXOR(rect, disk)
        >>> xor.set_sampling_config(create_config_from_edict(sampling_config_csg))
        >>> domain = xor.sampling(geom_type="domain")
        >>> bc, bc_normal = xor.sampling(geom_type="bc")
        >>> print(domain.shape)
        (1000, 2)
    """
    def __init__(self, geom1, geom2, sampling_config=None):
        """This class returns geom1 xor geom2"""
        _check_geom([geom1, geom2])
        name = geom1.name + "_xor_" + geom2.name
        self.columns_dict = {}
        max_coord_min = np.minimum(geom1.coord_min, geom2.coord_min)
        min_coord_max = np.maximum(geom1.coord_max, geom2.coord_max)
        super(CSGXOR, self).__init__(name, geom1, geom2, max_coord_min, min_coord_max, sampling_config)
        if sampling_config is None:
            sampling_config = geom1.sampling_config
        else:
            self.set_sampling_config(sampling_config)

    def _inside(self, points):
        """Check whether points in geom1 xor geom2 or not"""
        inside1 = self.geom1._inside(points)
        inside2 = self.geom2._inside(points)
        inside1_not_inside2 = np.logical_and(inside1, ~inside2)
        inside2_not_inside1 = np.logical_and(~inside1, inside2)
        return np.logical_or(inside1_not_inside2, inside2_not_inside1)

    def _random_domain_points(self):
        """Sample points in geom1 xor geom2"""
        xor_size = self.sampling_config.domain.size
        sampler = self.sampling_config.domain.sampler
        xor_domain_points = np.empty(shape=(xor_size, self.dim))

        index = 0
        while index < xor_size:
            domain_points = sample(xor_size, self.dim, sampler) * (self.coord_max - self.coord_min) + self.coord_min
            domain_points = np.reshape(domain_points, (-1, self.dim))
            xor_points = domain_points[self._inside(domain_points)]
            added_size = len(xor_points)
            xor_domain_points[index: min(xor_size, added_size + index)] = \
                xor_points[:min(added_size, xor_size - index)]
            index += added_size
        return xor_domain_points

    def _random_boundary_points(self):
        """Sample points in geom1 xor geom2"""
        xor_size = self.sampling_config.bc.size
        need_normal = self.sampling_config.bc.with_normal
        xor_points = np.empty(shape=(xor_size, self.dim))
        if need_normal:
            xor_normal = np.empty(shape=(xor_size, self.dim))

        index = 0
        if need_normal:
            while index < xor_size:
                boundary_from_geom1, normal_from_geom1 = self.geom1.sampling(geom_type="BC")
                boundary_from_geom2, normal_from_geom2 = self.geom2.sampling(geom_type="BC")
                index_in_geom1_out_geom2 = ~self.geom2._inside(boundary_from_geom1)
                index_in_geom1_in_geom2 = self.geom2._inside(boundary_from_geom1)
                index_in_geom2_out_geom1 = ~self.geom1._inside(boundary_from_geom2)
                index_in_geom2_in_geom1 = self.geom1._inside(boundary_from_geom2)

                boundary_from_geom1_out_geom2 = boundary_from_geom1[index_in_geom1_out_geom2]
                boundary_from_geom1_in_geom2 = boundary_from_geom1[index_in_geom1_in_geom2]
                boundary_from_geom2_out_geom1 = boundary_from_geom2[index_in_geom2_out_geom1]
                boundary_from_geom2_in_geom1 = boundary_from_geom2[index_in_geom2_in_geom1]
                boundary_from_csg = np.concatenate((boundary_from_geom1_out_geom2,
                                                    boundary_from_geom1_in_geom2,
                                                    boundary_from_geom2_out_geom1,
                                                    boundary_from_geom2_in_geom1), axis=0)
                normal_from_geom1_out_geom2 = normal_from_geom1[index_in_geom1_out_geom2]
                normal_from_geom1_in_geom2 = -1.0 * normal_from_geom1[index_in_geom1_in_geom2]
                normal_from_geom2_out_geom1 = normal_from_geom2[index_in_geom2_out_geom1]
                normal_from_geom2_in_geom1 = -1.0 * normal_from_geom2[index_in_geom2_in_geom1]
                normal_from_csg = np.concatenate((normal_from_geom1_out_geom2,
                                                  normal_from_geom1_in_geom2,
                                                  normal_from_geom2_out_geom1,
                                                  normal_from_geom2_in_geom1), axis=0)
                added_size = len(normal_from_csg)
                rand_index = np.random.permutation(added_size)
                boundary_from_csg = boundary_from_csg[rand_index]
                normal_from_csg = normal_from_csg[rand_index]
                xor_points[index: min(xor_size, added_size + index)] = \
                    boundary_from_csg[:min(added_size, xor_size - index)]
                xor_normal[index: min(xor_size, added_size + index)] = \
                    normal_from_csg[:min(added_size, xor_size - index)]
                index += added_size
            return xor_points, xor_normal
        while index < xor_size:
            boundary_from_geom1 = self.geom1.sampling(geom_type="BC")
            boundary_from_geom2 = self.geom2.sampling(geom_type="BC")
            boundary_from_csg = np.concatenate((boundary_from_geom1, boundary_from_geom2), axis=0)
            boundary_from_csg = np.random.permutation(boundary_from_csg)
            added_size = len(boundary_from_csg)
            xor_points[index: min(xor_size, added_size + index)] = \
                boundary_from_csg[:min(added_size, xor_size - index)]
            index += added_size
        return xor_points
