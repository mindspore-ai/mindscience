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
"""2d geometry"""

from __future__ import absolute_import
import numpy as np
import numpy.linalg as LA
from mindspore import log as logger

from .geometry_base import Geometry, DATA_TYPES, GEOM_TYPES, SamplingConfig
from .geometry_nd import HyperCube
from .geom_utils import sample, polar_sample, generate_mesh
from .shapes import adapter, simplex, pentagon
from ..utils.check_func import check_param_type, check_param_type_value

_SPACE = " "


class Disk(Geometry):
    r"""
    Definition of Disk object.

    Args:
        name (str): name of the disk.
        center (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]):
            center coordinates of the disk.
        radius (Union[int, float]): radius of the disk.
        dtype (numpy.dtype): data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``None``.

    Raises:
        ValueError: If `center` is neither list nor tuple of length 2.
        ValueError: If `radius` is negative.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, Disk
        >>> disk_mesh = dict({'domain': dict({'random_sampling': False, 'size' : [100, 180]}),
        ...                   'BC': dict({'random_sampling': False, 'size': 200, 'with_normal' : True,})})
        >>> disk = Disk("disk", (-1.0, 0), 2.0, sampling_config=generate_sampling_config(disk_mesh))
        >>> domain = disk.sampling(geom_type="domain")
        >>> bc, bc_normal = disk.sampling(geom_type="BC")
        >>> print(bc.shape)
        (200, 2)
    """
    def __init__(self, name, center, radius, dtype=np.float32, sampling_config=None):
        self.sampling_config = sampling_config
        check_param_type(center, "center", data_type=(np.ndarray, tuple, list))
        self.center = np.array(center)
        if len(self.center) != 2:
            raise ValueError("Disk: {}'s center position should be 2D array, but got {} with dim {}".format(
                name, self.center, len(self.center)))
        for ele in self.center:
            check_param_type(ele, "ele in center", data_type=DATA_TYPES, exclude_type=bool)

        check_param_type(radius, "radius", data_type=[int, float], exclude_type=bool)
        if radius <= 0:
            raise ValueError("Disk: {}'s radius should not be >=0.0, but got: {}".format(name, radius))
        self.radius = radius
        self.columns_dict = {}
        coord_min = self.center - self.radius
        coord_max = self.center + self.radius
        super(Disk, self).__init__(name, 2, coord_min, coord_max, dtype, sampling_config)

    def _inside(self, points, strict=False):
        """whether inside domain"""
        return LA.norm(points - self.center, axis=-1) < self.radius if strict \
            else LA.norm(points - self.center, axis=-1) <= self.radius

    def _on_boundary(self, points):
        """whether on domain boundary"""
        return np.isclose(LA.norm(points - self.center, axis=-1), self.radius)

    def _boundary_normal(self, points):
        """get the boundary normal vector"""
        points = points[self._on_boundary(points)]
        r = points - self.center
        r_norm = LA.norm(r, axis=-1, keepdims=True)
        return r / r_norm

    def _random_disk_boundary_points(self, need_normal=False):
        """Randomly generate boundary points"""
        size = self.sampling_config.bc.size
        sampler = self.sampling_config.bc.sampler
        theta = 2 * np.pi * sample(size, 1, sampler)
        circle_xy = np.hstack([np.cos(theta), np.sin(theta)])
        data = self.center + circle_xy * self.radius
        data = np.reshape(data, (-1, self.dim))
        if need_normal:
            normal_data = self._boundary_normal(data)
            normal_data = np.reshape(normal_data, (-1, self.dim))
            return data, normal_data
        return data

    def _random_disk_domain_points(self):
        """Randomly generate domain points"""
        size = self.sampling_config.domain.size
        sampler = self.sampling_config.domain.sampler
        r_theta = sample(size, 2, sampler)
        data = self.center + polar_sample(r_theta) * self.radius
        data = np.reshape(data, (-1, self.dim))
        return data

    def _grid_disk_boundary_points(self, need_normal=False):
        """Generate uniformly distributed domain points"""
        size = self.sampling_config.bc.size
        theta = np.linspace(0, 2 * np.pi, num=size, endpoint=False)
        cartesian = np.vstack((np.cos(theta), np.sin(theta))).T
        data = self.radius * cartesian + self.center
        data = np.reshape(data, (-1, self.dim))
        if need_normal:
            normal_data = self._boundary_normal(data)
            normal_data = np.reshape(normal_data, (-1, self.dim))
            return data, normal_data
        return data

    def _grid_disk_domain_points(self):
        """Generate uniformly distributed domain points"""
        mesh_size = self.sampling_config.domain.size
        if len(mesh_size) != self.dim:
            raise ValueError("For grid sampling, length of mesh_size list: {} should be equal to dimension: {}".format(
                mesh_size, self.dim
            ))
        r_theta_mesh = generate_mesh(np.array([0, 0]), np.array([1, 1]), mesh_size, endpoint=False)
        cartesian = np.zeros(r_theta_mesh.shape)
        cartesian[:, 0] = r_theta_mesh[:, 0] * self.radius * np.cos(2 * np.pi * r_theta_mesh[:, 1])
        cartesian[:, 1] = r_theta_mesh[:, 0] * self.radius * np.sin(2 * np.pi * r_theta_mesh[:, 1])
        data = cartesian + self.center
        data = np.reshape(data, (-1, self.dim))
        return data

    def _get_sdf(self, domain_points):
        """calculate sign distance function"""
        sdf = self.radius - np.linalg.norm(domain_points - self.center, axis=1)
        return -sdf

    def sampling(self, geom_type="domain"):
        """
        sampling domain and boundary points

        Args:
            geom_type (str): geometry type: can be ``'domain'`` or ``'BC'``. Default: ``'domain'``.

                - ``'domain'``, feasible domain of the problem.
                - ``'BC'``, boundary of the problem.

        Returns:
            Numpy.array. If the with_normal property of boundary configuration is true, returns 2D numpy array with
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
            logger.info("Sampling domain points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))
            column_name = self.name + "_domain_points"
            if config.domain.random_sampling:
                disk_data = self._random_disk_domain_points()
            else:
                disk_data = self._grid_disk_domain_points()
            self.columns_dict["domain"] = [column_name]
            disk_data = disk_data.astype(self.dtype)
            if config.domain.with_sdf:
                sdf = self._get_sdf(disk_data)
                sdf = sdf.astype(self.dtype)
                sdf_column_name = self.name + "_domain_sdf"
                self.columns_dict.get("domain").append(sdf_column_name)
                return disk_data, sdf
            return disk_data
        if geom_type.lower() == "bc":
            check_param_type(config.bc, _SPACE.join((self.geom_type, self.name, "'s bc config")),
                             exclude_type=type(None))
            logger.info("Sampling BC points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.bc))
            if config.bc.with_normal:
                if config.bc.random_sampling:
                    disk_data, disk_data_normal = self._random_disk_boundary_points(need_normal=True)
                else:
                    disk_data, disk_data_normal = self._grid_disk_boundary_points(need_normal=True)
                column_data = self.name + "_BC_points"
                column_normal = self.name + "_BC_normal"
                self.columns_dict["BC"] = [column_data, column_normal]
                disk_data = disk_data.astype(self.dtype)
                disk_data_normal = disk_data_normal.astype(self.dtype)
                return disk_data, disk_data_normal

            if config.bc.random_sampling:
                disk_data = self._random_disk_boundary_points(need_normal=False)
            else:
                disk_data = self._grid_disk_boundary_points(need_normal=False)
            column_data = self.name + "_BC_points"
            self.columns_dict["BC"] = [column_data]
            disk_data = disk_data.astype(self.dtype)
            return disk_data
        raise ValueError("Unknown geom_type: {}, only \"domain/BC\" are supported for {}:{}".format(
            geom_type, self.geom_type, self.name))


class Rectangle(HyperCube):
    r"""
    Definition of Rectangle object.

    Args:
        name (str): name of the rectangle.
        coord_min (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]):
            coordinates of the bottom left corner of rectangle.
        coord_max (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]):
            coordinates of the top right corner of rectangle.
        dtype (numpy.dtype): data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, Rectangle
        >>> rectangle_mesh = dict({'domain': dict({'random_sampling': False, 'size': [50, 25]}),
        ...                        'BC': dict({'random_sampling': False, 'size': 300, 'with_normal': True,}),})
        >>> rectangle = Rectangle("rectangle", (-3.0, 1), (1, 2),
        ...                       sampling_config=generate_sampling_config(rectangle_mesh))
        >>> domain = rectangle.sampling(geom_type="domain")
        >>> bc, bc_normal = rectangle.sampling(geom_type="BC")
        >>> print(domain.shape)
        (1250, 2)
    """
    def __init__(self, name, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(Rectangle, self).__init__(name, 2, coord_min, coord_max, dtype=dtype, sampling_config=sampling_config)


class Triangle(adapter.Geometry):
    r"""
    Definition of triangle object.

    Args:
        name (str): name of the triangle.
        vertices (numpy.ndarray): vertices of the triangle.
        boundary_type (str): this can be ``'uniform'`` or ``'unweighted'``. Default: ``'uniform'``.

            - ``'uniform'``, the expected number of samples in each boundary is proportional to the
              area (length) of the boundary.
            - ``'unweighted'``, the expected number of samples in each boundary is the same.

        dtype (numpy.dtype): data type of sampled point data type. Default: ``np.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``none``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, Triangle
        >>> triangle_mesh = dict({'domain': dict({'random_sampling': True, 'size': 300}),
        ...                       'BC': dict({'random_sampling': True, 'size': 300, 'with_normal': False,}),})
        >>> vertices = np.array([[0., .1], [.9, .2], [.5, .6]])
        >>> triangle = Triangle("triangle", vertices,
        ...                     sampling_config=generate_sampling_config(triangle_mesh))
        >>> domain = triangle.sampling(geom_type="domain")
        >>> bc = triangle.sampling(geom_type="bc")
        >>> print(domain.shape)
        (300, 2)
    """
    def __init__(self, name, vertices,
                 boundary_type="uniform", dtype=np.float32, sampling_config=None):
        super(Triangle, self).__init__(
            name=name,
            shape=simplex.Simplex(vertices, boundary_type),
            dim=2,
            coord_min=np.min(vertices, axis=0),
            coord_max=np.max(vertices, axis=0),
            dtype=dtype,
            sampling_config=sampling_config,
        )


class Pentagon(adapter.Geometry):
    r"""
    Definition of pentagon object.

    Args:
        name (str): name of the pentagon.
        vertices (numpy.ndarray): vertices of the pentagon in an anti-clockwise order.
        boundary_type (str): this can be ``'uniform'`` or ``'unweighted'``. Default: ``'uniform'``.

            - ``'uniform'``, the expected number of samples in each boundary is proportional to the
              area (length) of the boundary.
            - ``'unweighted'``, the expected number of samples in each boundary is the same.

        dtype (numpy.dtype): data type of sampled point data type. Default: ``np.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``none``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, Pentagon
        >>> pentagon_mesh = dict({'domain': dict({'random_sampling': True, 'size': 300}),
        ...                       'BC': dict({'random_sampling': True, 'size': 300, 'with_normal': False,}),})
        >>> vertices = np.array([[0., .1], [.5, .1], [.9, .2], [.7, .6], [.2, .5]])
        >>> pentagon = Pentagon("pentagon", vertices,
        ...                     sampling_config=generate_sampling_config(pentagon_mesh))
        >>> domain = pentagon.sampling(geom_type="domain")
        >>> bc = pentagon.sampling(geom_type="bc")
        >>> print(domain.shape)
        (300, 2)
    """
    def __init__(self, name, vertices,
                 boundary_type="uniform", dtype=np.float32, sampling_config=None):
        super(Pentagon, self).__init__(
            name=name,
            shape=pentagon.Pentagon(vertices, boundary_type),
            dim=2,
            coord_min=np.min(vertices, axis=0),
            coord_max=np.max(vertices, axis=0),
            dtype=dtype,
            sampling_config=sampling_config,
        )


class Polygon(Geometry):
    pass
