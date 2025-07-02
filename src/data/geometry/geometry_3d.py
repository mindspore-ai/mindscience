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
import numpy

from .geometry_base import Geometry
from .geometry_nd import HyperCube
from .shapes import adapter, rotating, simplex


class Cuboid(HyperCube):
    r"""
    Definition of Cuboid object.

    Args:
        name (str): name of the cuboid.
        coord_min (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]):
            coordinates of the bottom left back corner of cuboid.
        coord_max (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]):
            coordinates of the top right front corner of cuboid.
        dtype (numpy.dtype): data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, Cuboid
        >>> cuboid_mesh = dict({'domain': dict({'random_sampling': False, 'size': [50, 50, 25]}),
        ...                      'BC': dict({'random_sampling': False, 'size': 1000, 'with_normal': True,}),})
        >>> cuboid = Cuboid("cuboid", (-3.0, 1, 0), (1, 2, 1), sampling_config=generate_sampling_config(cuboid_mesh))
        >>> domain = cuboid.sampling(geom_type="domain")
        >>> bc, bc_normal = cuboid.sampling(geom_type="BC")
        >>> print(domain.shape)
        (62500, 3)
    """
    def __init__(self, name, coord_min, coord_max, dtype=numpy.float32, sampling_config=None):
        super(Cuboid, self).__init__(name, 3, coord_min, coord_max, dtype=dtype, sampling_config=sampling_config)


class Tetrahedron(adapter.Geometry):
    r"""
    Definition of tetrahedron object.

    Args:
        name (str): name of the tetrahedron.
        vertices (numpy.ndarray): vertices of the tetrahedron.
        boundary_type (str): this can be ``'uniform'`` or ``'unweighted'``. Default: ``'uniform'``.

            - ``'uniform'``, the expected number of samples in each boundary is proportional to the
              area (length) of the boundary.
            - ``'unweighted'``, the expected number of samples in each boundary is the same.

        dtype (numpy.dtype): data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``none``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.geometry import generate_sampling_config, Tetrahedron
        >>> tetrahedron_mesh = dict({'domain': dict({'random_sampling': True, 'size': 300}),
        ...                          'BC': dict({'random_sampling': True, 'size': 300, 'with_normal': False,}),})
        >>> vertices = np.array([[0., .1, 0.], [.9, .2, .1], [.5, .6, 0.1], [.6, .5, .8]])
        >>> tetrahedron = Tetrahedron("tetrahedron", vertices,
        ...                           sampling_config=generate_sampling_config(tetrahedron_mesh))
        >>> domain = tetrahedron.sampling(geom_type="domain")
        >>> bc = tetrahedron.sampling(geom_type="bc")
        >>> print(domain.shape)
        (300, 2)
    """
    def __init__(self, name, vertices,
                 boundary_type="uniform", dtype=numpy.float32, sampling_config=None):
        super(Tetrahedron, self).__init__(
            name=name,
            shape=simplex.Simplex(vertices, boundary_type),
            dim=3,
            coord_min=numpy.min(vertices, axis=0),
            coord_max=numpy.max(vertices, axis=0),
            dtype=dtype,
            sampling_config=sampling_config
        )


class Sphere(Geometry):
    pass


class Cylinder(adapter.Geometry):
    r"""
    Definition of cylinder object.

    Args:
        name (str): name of the cylinder.
        centre (numpy.ndarray): origin of the bottom disk.
        radius (float): Radius of the cylinder.
        h_min (float): Height coordinate of the bottom disk.
        h_max (float): Height coordinate of the top disk.
        h_axis (int): Axis of the normal vector of the bottom disk.
        boundary_type (str): this can be ``'uniform'`` or ``'unweighted'``. Default: ``'uniform'``.

            - ``'uniform'``, the expected number of samples in each boundary is proportional to the
              area (length) of the boundary.
            - ``'unweighted'``, the expected number of samples in each boundary is the same.

        dtype (numpy.dtype): data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``none``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.geometry import generate_sampling_config, Cylinder
        >>> cylinder_mesh = dict({'domain': dict({'random_sampling': True, 'size': 300}),
        ...                       'BC': dict({'random_sampling': True, 'size': 300, 'with_normal': False,}),})
        >>> vertices = np.array([[0., .1, 0.], [.9, .2, .1], [.5, .6, 0.1], [.6, .5, .8]])
        >>> centre = np.array([0., 0.5])
        >>> radius = 1.5
        >>> h_min = -7.
        >>> h_max = 7.
        >>> h_axis = 2
        >>> cylinder = Cylinder("cylinder", centre, radius, h_min, h_max, h_axis,
        ...                     sampling_config=generate_sampling_config(cylinder_mesh))
        >>> domain = cylinder.sampling(geom_type="domain")
        >>> bc = cylinder.sampling(geom_type="bc")
        >>> print(domain.shape)
        (300, 2)
    """
    def __init__(self, name, centre, radius, h_min, h_max, h_axis,
                 boundary_type="uniform", dtype=numpy.float32, sampling_config=None):

        shape = rotating.Cylinder(centre, radius, h_min, h_max, h_axis, boundary_type)
        coord_min = numpy.append(numpy.asarray(centre) - numpy.asarray(radius), h_min)
        coord_max = numpy.append(numpy.asarray(centre) + numpy.asarray(radius), h_max)
        super(Cylinder, self).__init__(
            name=name,
            shape=shape,
            dim=3,
            coord_min=coord_min,
            coord_max=coord_max,
            dtype=dtype,
            sampling_config=sampling_config
        )


class Cone(adapter.Geometry):
    r"""
    Definition of cone object.

    Args:
        name (str): name of the cone.
        centre (numpy.ndarray): origin of the bottom disk.
        radius (float): Radius of the bottom disk.
        h_min (float): Height coordinate of the bottom disk.
        h_max (float): Maximum Height coordinate of the cone.
        h_axis (int): Axis of the normal vector of the bottom disk.
        boundary_type (str): this can be ``'uniform'`` or ``'unweighted'``. Default: ``'uniform'``.

            - ``'uniform'``, the expected number of samples in each boundary is proportional to the
              area (length) of the boundary.
            - ``'unweighted'``, the expected number of samples in each boundary is the same.

        dtype (numpy.dtype): data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``none``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.geometry import generate_sampling_config, Cone
        >>> cone_mesh = dict({'domain': dict({'random_sampling': True, 'size': 300}),
        ...                   'BC': dict({'random_sampling': True, 'size': 300, 'with_normal': False,}),})
        >>> vertices = np.array([[0., .1, 0.], [.9, .2, .1], [.5, .6, 0.1], [.6, .5, .8]])
        >>> centre = np.array([0., 0.5])
        >>> radius = 1.5
        >>> h_min = -7.
        >>> h_max = 7.
        >>> h_axis = 2
        >>> cone = Cone("cone", centre, radius, h_min, h_max, h_axis,
        ...             sampling_config=generate_sampling_config(cone_mesh))
        >>> domain = cone.sampling(geom_type="domain")
        >>> bc = cone.sampling(geom_type="bc")
        >>> print(domain.shape)
        (300, 2)
    """
    def __init__(self, name, centre, radius, h_min, h_max, h_axis,
                 boundary_type="uniform", dtype=numpy.float32, sampling_config=None):

        shape = rotating.Cone(centre, radius, h_min, h_max, h_axis, boundary_type)
        coord_min = numpy.append(numpy.asarray(centre) - numpy.asarray(radius), h_min)
        coord_max = numpy.append(numpy.asarray(centre) + numpy.asarray(radius), h_max)
        super(Cone, self).__init__(
            name=name,
            shape=shape,
            dim=3,
            coord_min=coord_min,
            coord_max=coord_max,
            dtype=dtype,
            sampling_config=sampling_config
        )
