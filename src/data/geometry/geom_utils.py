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
"""utils for geometry"""
from __future__ import absolute_import
import numpy as np
from scipy.stats import qmc

from .geometry_base import PartSamplingConfig, SamplingConfig, GEOM_TYPES, SAMPLER_TYPES
from ..utils.check_func import check_param_type


def generate_sampling_config(dict_config):
    """
    Convert from dict to SamplingConfig.

    Args:
        dict_config (dict): dict containing configuration info.

    Returns:
        geometry_base.SamplingConfig, sampling configuration.

    Raises:
        ValueError: If part_dict_config can not be generated from input dict.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config
        >>> rect_config = dict({
        ...     'domain': dict({
        ...         'random_sampling': True,
        ...         'size': 200,
        ...         'with_sdf': False,
        ...     }),
        ...     'BC': dict({
        ...         'random_sampling': True,
        ...         'size': 50,
        ...         'with_normal': True,
        ...     })
        ... })
        >>> sampling_config = generate_sampling_config(rect_config)
    """
    check_param_type(dict_config, "dict_config", data_type=dict)
    part_dict_config = {}
    for geom_type in dict_config.keys():
        if geom_type in GEOM_TYPES and dict_config[geom_type]:
            part_dict_config[geom_type] = PartSamplingConfig(dict_config[geom_type].get("size", 1),
                                                             dict_config[geom_type].get("random_sampling", True),
                                                             dict_config[geom_type].get("sampler", "uniform"),
                                                             dict_config[geom_type].get("random_merge", True),
                                                             dict_config[geom_type].get("with_normal", False),
                                                             dict_config[geom_type].get("with_sdf", False))
    if part_dict_config:
        return SamplingConfig(part_dict_config)
    raise ValueError("Unknown sampling info, please check your config")


_sampler_method = {
    "lhs": qmc.LatinHypercube,
    "halton": qmc.Halton,
    "sobol": qmc.Sobol,
    "uniform": np.random.rand
}


def sample(size, dimension, sampler="uniform"):
    """function for sampling points by different random methods"""
    sampler = sampler.lower()
    if sampler not in _sampler_method.keys() or sampler not in SAMPLER_TYPES:
        raise ValueError("Unknown sampler method {}, only {} are supported".format(sampler, _sampler_method.keys()))
    sample_method = _sampler_method.get(sampler)
    if sampler == "uniform":
        data = sample_method(size, dimension)
    else:
        data = sample_method(d=dimension).random(size)
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return data


def polar_sample(r_theta):
    """convert polar coordinate system to rectangle coordinate system"""
    r, theta = r_theta[:, 0], 2 * np.pi * r_theta[:, 1]
    coord_xy = np.sqrt(r) * np.vstack([np.cos(theta), np.sin(theta)])
    return coord_xy.T


def generate_mesh(coord_min, coord_max, mesh_size, endpoint=True):
    """generate regularly distributed mesh"""
    dimension = len(coord_min)
    if dimension != len(coord_max) or dimension != len(mesh_size):
        raise ValueError("Inconsistent dimension info, coord_min: {}, coord_max: {}, mesh_size: {}"
                         .format(coord_min, coord_max, mesh_size))

    axis_x = np.linspace(coord_min[0], coord_max[0], mesh_size[0], endpoint=endpoint)
    mesh = None
    if dimension == 1:
        mesh = axis_x[:, np.newaxis].astype(np.float32)
        return mesh

    axis_y = np.linspace(coord_min[1], coord_max[1], mesh_size[1], endpoint=endpoint)
    if dimension == 2:
        mesh_x, mesh_y = np.meshgrid(axis_x, axis_y)
        mesh = np.hstack((mesh_x.flatten()[:, None], mesh_y.flatten()[:, None])).astype(np.float32)
        return mesh

    axis_z = np.linspace(coord_min[2], coord_max[2], mesh_size[2], endpoint=endpoint)
    if dimension == 3:
        mesh_x, mesh_y, mesh_z = np.meshgrid(axis_x, axis_y, axis_z)
        mesh = np.hstack((mesh_x.flatten()[:, None], mesh_y.flatten()[:, None],
                          mesh_z.flatten()[:, None])).astype(np.float32)
        return mesh

    axis_t = np.linspace(coord_min[3], coord_max[3], mesh_size[3], endpoint=endpoint)
    if dimension == 4:
        mesh_x, mesh_y, mesh_z, mesh_t = np.meshgrid(axis_x, axis_y, axis_z, axis_t)
        mesh = np.hstack((mesh_x.flatten()[:, None], mesh_y.flatten()[:, None],
                          mesh_z.flatten()[:, None], mesh_t.flatten()[:, None])).astype(np.float32)
        return mesh
    if dimension > 4:
        raise ValueError("Only dimension <= 4 are supported, but got {}".format(dimension))
    return mesh
