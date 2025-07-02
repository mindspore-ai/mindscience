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
"""3d geometry"""

from __future__ import absolute_import

import numpy as np
from mindspore import log as logger

from .geometry_base import Geometry, GEOM_TYPES, SamplingConfig
from .geom_utils import sample, generate_mesh
from ..utils.check_func import check_param_type, check_param_type_value

_SPACE = " "


class FixedPoint(Geometry):
    r"""
    Definition of fixed point object.

    Args:
        name (str): name of the fixed point.
        coord (Union[int, float, tuple, list, numpy.ndarray]): coordinate of the fixed point. if the parameter type is
            tuple or list, the element support tuple[int, int], tuple[float, float], list[int, int], list[float, float].
        dtype (numpy.dtype): Data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, FixedPoint
        >>> hypercube_random = dict({
        ...      'domain': dict({
        ...          'random_sampling': True,
        ...          'size': 1,
        ...          'sampler': 'uniform'
        ...         })
        ...  })
        >>> sampling_config = generate_sampling_config(hypercube_random)
        >>> point = FixedPoint("FixedPoint", [-1, 2, 1], sampling_config=sampling_config)
        >>> domain = point.sampling(geom_type="domain")
        >>> print(domain.shape)
        (1, 3)
    """
    def __init__(self, name, coord, dtype=np.float32, sampling_config=None):
        if isinstance(coord, (int, float)):
            coord = [coord]
        super(FixedPoint, self).__init__(name, len(coord), coord, coord, dtype, sampling_config)
        self.coord = coord
        self.columns_dict = {}
        self.length = 0.0
        self.vol = 0.0
        self.area = 0.0

    def _inside(self, points, strict=False):
        raise NotImplementedError("{}._inside not implemented".format(self.geom_type))

    def _on_boundary(self, points):
        raise NotImplementedError("{}._on_boundary not implemented".format(self.geom_type))

    def _boundary_normal(self, points):
        raise NotImplementedError("{}._boundary_normal not implemented".format(self.geom_type))

    def sampling(self, geom_type="domain"):
        """
        sampling points

        Args:
            geom_type (str): geometry type, which supports ``'domain'`` and ``'BC'``. Default: ``'domain'``.

        Returns:
            Numpy.ndarray, 2D numpy array with or without boundary normal vectors.

        Raises:
            ValueError: If `config` is ``None``.
            KeyError: If `geom_type` is ``'domain'`` but `config.domain` is ``None``.
            KeyError: If `geom_type` is ``'BC'`` but `config.bc` is ``None``.
            ValueError: If `geom_type` is neither ``'BC'`` nor ``'domain'``.
        """
        config = self.sampling_config
        check_param_type_value(geom_type, _SPACE.join((self.geom_type, self.name, "'s geom_type")),
                               GEOM_TYPES, data_type=str)
        if geom_type.lower() == "domain":
            logger.info("Sampling domain points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))
            column_name = self.name + "_domain_points"
            data = np.tile(self.coord, (self.sampling_config.domain.size, 1))
            data = np.reshape(data, (-1, self.dim))
            self.columns_dict["domain"] = [column_name]
            data = data.astype(self.dtype)
            return data
        if geom_type.lower() == "bc":
            logger.info("Sampling BC points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.bc))
            if config.bc.with_normal:
                raise ValueError("Normal is not supported on point: {}")
            data = np.tile(self.coord, (self.sampling_config.bc.size, 1))
            data = np.reshape(data, (-1, self.dim))
            column_data = self.name + "_BC_points"
            self.columns_dict["BC"] = [column_data]
            data = data.astype(self.dtype)
            return data
        raise ValueError("Unknown geom_type: {}, only \"domain/BC\" are supported for {}:{}".format(
            geom_type, self.geom_type, self.name))


class HyperCube(Geometry):
    r"""
    Definition of HyperCube object.

    Args:
        name (str): name of the hyper cube.
        dim (int): number of dimensions.
        coord_min (Union[int, float, tuple, list, numpy.ndarray]): minimal coordinate of the hyper cube. if the
            parameter type is tuple or list, the element support tuple[int, int], tuple[float, float], list[int, int],
            list[float, float].
        coord_max (Union[int, float, tuple, list, numpy.ndarray]): maximal coordinate of the hyper cube. if the
            parameter type is tuple or list, the element support tuple[int, int], tuple[float, float], list[int, int],
            list[float, float].
        dtype (numpy.dtype): Data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``None``.

    Raises:
        TypeError: `sampling_config` is not instance of class SamplingConfig.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, HyperCube
        >>> hypercube_random = dict({
        ...      'domain': dict({
        ...          'random_sampling': True,
        ...          'size': 1000,
        ...          'sampler': 'uniform'
        ...         }),
        ...      'BC': dict({
        ...          'random_sampling': True,
        ...          'size': 200,
        ...          'sampler': 'uniform',
        ...          'with_normal': False,
        ...      }),
        ...  })
        >>> sampling_config = generate_sampling_config(hypercube_random)
        >>> hypercube = HyperCube("HyperCube", 3, [-1, 2, 1], [0, 3, 2], sampling_config=sampling_config)
        >>> domain = hypercube.sampling(geom_type="domain")
        >>> bc = hypercube.sampling(geom_type="BC")
        >>> print(domain.shape)
        (1000, 3)
    """
    def __init__(self, name, dim, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(HyperCube, self).__init__(name, dim, coord_min, coord_max, dtype, sampling_config)
        if np.any(self.coord_max - self.coord_min <= 0.0):
            raise ValueError("coord_min should be smaller than coord_max, but got coord_min: {}, coord_max: {}".format(
                self.coord_min, self.coord_max))
        self.columns_dict = {}
        self.length = self.coord_max - self.coord_min
        self.vol = np.prod(self.length)
        area = 0
        for i in range(dim):
            area += self.vol / self.length[i]
        self.area = area * 2.0

    def _inside(self, points, strict=False):
        """whether inside domain"""
        valid_min = (self.coord_min < points).all(axis=-1) if strict else (self.coord_min <= points).all(axis=-1)
        valid_max = (self.coord_max > points).all(axis=-1) if strict else (self.coord_max >= points).all(axis=-1)
        return np.logical_and(valid_min, valid_max)

    def _on_boundary(self, points):
        """whether on geometry's boundary"""
        near_boundary = np.logical_or(np.any(np.isclose(points, self.coord_min), axis=-1),
                                      np.any(np.isclose(points, self.coord_max), axis=-1))
        return near_boundary

    def _boundary_normal(self, points):
        """get the normal vector of boundary points"""
        points = self._filter_corner_points(points)
        normal = np.isclose(points, self.coord_min) * -1.0 + np.isclose(points, self.coord_max) * 1.0
        return normal

    def _filter_corner_points(self, points):
        corner = np.isclose(points, self.coord_min) + np.isclose(points, self.coord_max)
        have_corner = np.count_nonzero(corner, axis=-1)
        not_corner_points = np.where(have_corner <= 1)[0]
        return points[not_corner_points]

    def _random_domain_points(self):
        """randomly generate domain points"""
        size = self.sampling_config.domain.size
        sampler = self.sampling_config.domain.sampler
        data = sample(size, self.dim, sampler) * (self.coord_max - self.coord_min) + self.coord_min
        data = np.reshape(data, (-1, self.dim))
        return data

    def _grid_domain_points(self):
        """generate domain mesh points"""
        mesh_size = self.sampling_config.domain.size
        if len(mesh_size) != self.dim:
            raise ValueError("For grid sampling, length of mesh_size list: {} should be equal to dimension: {}".format(
                mesh_size, self.dim
            ))
        mesh_x = generate_mesh(self.coord_min, self.coord_max, mesh_size)
        data = np.reshape(mesh_x, (-1, self.dim))
        return data

    def _random_boundary_points(self, need_normal=False):
        """get boundary points randomly"""
        size = self.sampling_config.bc.size
        sampler = self.sampling_config.bc.sampler
        boundary_points = []
        for i in range(self.dim):
            area_i = self.vol / self.length[i]
            ratio = area_i * 2 / self.area
            num_sample = int(size * ratio)
            temp_boundary_points = sample(num_sample, self.dim, sampler)
            temp_boundary_points[np.arange(num_sample), i] = np.round(temp_boundary_points[np.arange(num_sample), i])
            boundary_points.extend([temp_boundary_points])
        data = np.concatenate(boundary_points, axis=0)
        data = np.random.permutation(data)
        data = data * (self.coord_max - self.coord_min) + self.coord_min
        if need_normal:
            data = self._filter_corner_points(data)
            data = np.reshape(data, (-1, self.dim))
            data_normal = self._boundary_normal(data)
            data_normal = np.reshape(data_normal, (-1, self.dim))
            return data, data_normal
        data = np.reshape(data, (-1, self.dim))
        return data

    def _grid_boundary_points(self, need_normal=False):
        """get gird boundary points"""
        size = self.sampling_config.bc.size
        if self.dim == 1:
            ds = self.area / 5.0
        else:
            ds = (self.area / size) ** (1 / (self.dim - 1))
        mesh_size = (self.length / ds).astype(np.int64)
        domain_data = generate_mesh(self.coord_min, self.coord_max, mesh_size)
        bound_cond = np.where(self._on_boundary(domain_data))[0]
        data = domain_data[bound_cond]
        data = np.reshape(data, (-1, self.dim))
        data = np.random.permutation(data)
        if need_normal:
            data = self._filter_corner_points(data)
            data_normal = self._boundary_normal(data)
            data_normal = np.reshape(data_normal, (-1, self.dim))
            return data, data_normal
        return data

    def _get_sdf(self, domain_points):
        """calculate sign distance function"""
        center = (self.coord_min + self.coord_max) / 2.0
        p_dist = np.abs(domain_points - center) - (self.coord_max - center)
        sdf = np.linalg.norm(np.maximum(p_dist, 0.0)) + np.minimum(np.max(p_dist, axis=1), 0.0)
        return -sdf

    def sampling(self, geom_type="domain"):
        """
        sampling points

        Args:
            geom_type (str): geometry type: can be ``'domain'`` or ``'BC'``. Default: ``'domain'``.

                - ``'domain'``, feasible domain of the problem.
                - ``'BC'``, boundary of the problem.

        Returns:
            Numpy.ndarray, if the with_normal property of boundary configuration is true, returns 2D numpy array with
                         boundary normal vectors. Otherwise, returns 2D numpy array without boundary normal vectors.

        Raises:
            ValueError: If `config` is ``None``.
            KeyError: If `geom_type` is ``'domain'`` but `config.domain` is ``None``.
            KeyError: If `geom_type` is ``'BC'`` but `config.bc` is ``None``.
            ValueError: If `geom_type` is neither ``'BC'`` nor ``'domain'``.
        """
        config = self.sampling_config
        check_param_type(config, _SPACE.join((self.geom_type, self.name, "'s sampling_config")),
                         data_type=SamplingConfig)
        check_param_type_value(geom_type, _SPACE.join((self.geom_type, self.name, "'s geom_type")),
                               GEOM_TYPES, data_type=str)
        if geom_type.lower() == "domain":
            check_param_type(config.domain, _SPACE.join((self.geom_type, self.name, "'s domain config")),
                             exclude_type=type(None))
            if config.domain is None:
                raise KeyError("Sampling config for domain of {}:{} should not be none"
                               .format(self.geom_type, self.name))
            logger.info("Sampling domain points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))
            column_name = self.name + "_domain_points"
            if config.domain.random_sampling:
                data = self._random_domain_points()
            else:
                data = self._grid_domain_points()
            self.columns_dict["domain"] = [column_name]
            data = data.astype(self.dtype)

            if config.domain.with_sdf:
                sdf = self._get_sdf(data)
                sdf = sdf.astype(self.dtype)
                sdf_column_name = self.name + "_domain_sdf"
                self.columns_dict.get("domain").append(sdf_column_name)
                return data, sdf
            return data
        if geom_type.lower() == "bc":
            check_param_type(config.bc, _SPACE.join((self.geom_type, self.name, "'s bc config")),
                             exclude_type=type(None))
            logger.info("Sampling BC points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.bc))
            if config.bc.with_normal:
                if config.bc.random_sampling:
                    data, data_normal = self._random_boundary_points(need_normal=True)
                else:
                    data, data_normal = self._grid_boundary_points(need_normal=True)
                column_data = self.name + "_BC_points"
                column_normal = self.name + "_BC_normal"
                self.columns_dict["BC"] = [column_data, column_normal]
                data = data.astype(self.dtype)
                data_normal = data_normal.astype(self.dtype)
                return data, data_normal

            if config.bc.random_sampling:
                data = self._random_boundary_points(need_normal=False)
            else:
                data = self._grid_boundary_points(need_normal=False)
            column_data = self.name + "_BC_points"
            self.columns_dict["BC"] = [column_data]
            data = data.astype(self.dtype)
            return data
        raise ValueError("Unknown geom_type: {}, only \"domain/BC\" are supported for {}:{}".format(
            geom_type, self.geom_type, self.name))
