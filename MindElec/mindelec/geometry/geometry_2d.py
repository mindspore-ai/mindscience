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

from .geometry_base import Geometry, DATA_TYPES, GEOM_TYPES
from .geometry_nd import HyperCube
from .utils import sample, polar_sample, generate_mesh


class Disk(Geometry):
    r"""
    Definition of Disk object.

    Args:
        name (str): name of the disk.
        center (Union[tuple[int, float], list[int, float], numpy.ndarray]): center coordinates of the disk.
        radius (Union[int, float]): radius of the disk.
        dtype (numpy.dtype): data type of sampled point data type. Default: numpy.float32.
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Raises:
        ValueError: If `center` is neither list nor tuple of length 2.
        ValueError: If `radius` is negative.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Disk
        >>> disk_mesh = edict({'domain': edict({'random_sampling': False, 'size' : [100, 180]}),
        ...                    'BC': edict({'random_sampling': False, 'size': 200, 'with_normal' : True,})})
        >>> disk = Disk("disk", (-1.0, 0), 2.0, sampling_config=create_config_from_edict(disk_mesh))
        >>> domain = disk.sampling(geom_type="domain")
        >>> bc, bc_normal = disk.sampling(geom_type="BC")
        >>> print(bc.shape)
        (200, 2)
    """
    def __init__(self, name, center, radius, dtype=np.float32, sampling_config=None):
        self.sampling_config = sampling_config
        if not isinstance(center, (np.ndarray, tuple, list)):
            raise TypeError("Disk: {}'s center should be tuple or list, but got: {}, type: {}".format(
                name, center, type(center)))
        self.center = np.array(center)
        if len(self.center) != 2:
            raise ValueError("Disk: {}'s center should be 2D array, but got: {}, dim: {}".format(
                name, self.center, len(self.center)))
        for ele in self.center:
            if not isinstance(ele, DATA_TYPES):
                raise TypeError("data type of center should be int/float, but got: {}, type: {}".format(
                    self.center, type(ele)
                ))
        if not isinstance(radius, (int, float)) or isinstance(radius, bool):
            raise TypeError("data type of radius should be int/float, but got: {}, type: {}".format(
                radius, type(radius)
            ))
        if radius <= 0:
            raise ValueError("Disk: {}'s radius should not be non-positive, but got: {}".format(name, radius))
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

    def sampling(self, geom_type="domain"):
        """
        sampling domain and boundary points

        Args:
            geom_type (str): geometry type

        Returns:
            Numpy.array, 2D numpy array with or without boundary normal vectors

        Raises:
            ValueError: If `config` is None.
            KeyError: If `geom_type` is `domain` but `config.domain` is None.
            KeyError: If `geom_type` is `BC` but `config.bc` is None.
            ValueError: If `geom_type` is neither `BC` nor `domain`.
        """
        config = self.sampling_config
        if config is None:
            raise ValueError("Sampling config for {}:{} is None, please call set_sampling_config method to set".format(
                self.geom_type, self.name))
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
            if config.domain.random_sampling:
                disk_data = self._random_disk_domain_points()
            else:
                disk_data = self._grid_disk_domain_points()
            self.columns_dict["domain"] = [column_name]
            disk_data = disk_data.astype(self.dtype)
            return disk_data
        if geom_type.lower() == "bc":
            if config.bc is None:
                raise KeyError("Sampling config for BC of {}:{} should not be none".format(self.geom_type, self.name))
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
        coord_min (Union[tuple[int, float], list[int, float], numpy.ndarray]): coordinates of the bottom
            left corner of rectangle.
        coord_max (Union[tuple[int, float], list[int, float], numpy.ndarray]): coordinates of the top
            right corner of rectangle.
        dtype (numpy.dtype): data type of sampled point data type. Default: numpy.float32.
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Rectangle
        >>> rectangle_mesh = edict({'domain': edict({'random_sampling': False, 'size': [50, 25]}),
        ...                         'BC': edict({'random_sampling': False, 'size': 300, 'with_normal': True,}),})
        >>> rectangle = Rectangle("rectangle", (-3.0, 1), (1, 2),
        ...                       sampling_config=create_config_from_edict(rectangle_mesh))
        >>> domain = rectangle.sampling(geom_type="domain")
        >>> bc, bc_normal = rectangle.sampling(geom_type="BC")
        >>> print(domain.shape)
        (1250, 2)
    """
    def __init__(self, name, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(Rectangle, self).__init__(name, 2, coord_min, coord_max, dtype=dtype, sampling_config=sampling_config)


class Triangle(Geometry):
    pass

class Polygon(Geometry):
    pass
