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
"""geometry with time domain"""

from __future__ import absolute_import
import copy
import numpy as np

from mindspore import log as logger

from .geometry_base import Geometry, SamplingConfig, GEOM_TYPES
from .geometry_1d import Interval
from ..utils.check_func import check_param_type, check_param_type_value

_space = " "


class TimeDomain(Interval):
    """
    Definition of Time Domain.

    Args:
        name (str): name of the time domain.
        start (Union[int, float]): start of the time domain. Default: ``0.0``.
        end (Union[int, float]): end of the time domain. Default: ``1.0``.
        dtype (numpy.dtype): Data type of sampled point data type. Default: ``numpy.float32``.
        sampling_config (SamplingConfig): sampling configuration. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, TimeDomain
        >>> time_config = dict({
        ...     'domain': dict({
        ...         'random_sampling': True,
        ...         'size': 100,
        ...         'sampler': 'lhs'
        ...     })
        ... })
        >>> time_domain = TimeDomain("time", 0.0, 1.0, sampling_config=generate_sampling_config(time_config))
        >>> domain = time_domain.sampling(geom_type="domain")
        >>> print(domain.shape)
        (100, 1)
    """

    def __init__(self, name, start=0.0, end=1.0, dtype=np.float32, sampling_config=None):
        self.start = start
        self.end = end
        super(TimeDomain, self).__init__(name, coord_min=start, coord_max=end, dtype=dtype,
                                         sampling_config=sampling_config)


class GeometryWithTime(Geometry):
    """
    Definition of geometry with time.

    Args:
        geometry (Geometry): geometry.
        timedomain (TimeDomain): time domain.
        sampling_config (SamplingConfig): sampling configuration. Default: ``None``.

    Raises:
        ValueError: If `sampling_config` is not ``None`` but `sampling_config.time` is ``None`` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.geometry import generate_sampling_config, Rectangle, TimeDomain, GeometryWithTime
        >>> rect_with_time_config = dict({
        ...     'domain': dict({
        ...         'random_sampling': True,
        ...         'size': 200,
        ...     }),
        ...     'BC': dict({
        ...         'random_sampling': False,
        ...         'size': 100,
        ...         'with_normal': True,
        ...     }),
        ...     'IC': dict({
        ...         'random_sampling': False,
        ...         'size': [10, 10],
        ...     }),
        ...     'time': dict({
        ...         'random_sampling': True,
        ...         'size': 10,
        ...     })
        ... })
        >>> rect = Rectangle("rect", [-1.0, -0.5], [1.0, 0.5])
        >>> time = TimeDomain("time", 0.0, 1.0)
        >>> rect_with_time = GeometryWithTime(rect, time)
        >>> sampling_config = generate_sampling_config(rect_with_time_config)
        >>> rect_with_time.set_sampling_config(sampling_config)
        >>> bc, bc_normal = rect_with_time.sampling(geom_type="BC")
        >>> domain = rect_with_time.sampling(geom_type="domain")
        >>> ic = rect_with_time.sampling(geom_type="IC")
        >>> print(domain.shape)
        (200, 3)
        >>> print(bc.shape)
        (90, 3)
        >>> print(ic.shape)
        (100, 3)
    """
    def __init__(self, geometry, timedomain, sampling_config=None):
        check_param_type(geometry, "geometry", data_type=Geometry)
        check_param_type(timedomain, "timedomain", data_type=TimeDomain)
        geometry = copy.deepcopy(geometry)
        timedomain = copy.deepcopy(timedomain)
        name = timedomain.name + "_" + geometry.name
        super(GeometryWithTime, self).__init__(name, geometry.dim, geometry.coord_min, geometry.coord_max,
                                               geometry.dtype)
        self.geom = geometry
        self.td = timedomain
        self.columns_dict = {}
        if not sampling_config:
            if self.geom.sampling_config and self.td.sampling_config:
                self.sampling_config = copy.deepcopy(self.geom.sampling_config)
                if not self.geom.sampling_config.domain or not self.td.sampling_config.domain:
                    logger.info("Undefined sampling info for {}:{}, please call set_sampling_config method to "
                                "reset sampling info".format(self.geom_type, self.name))
                    self.sampling_config = None
                else:
                    self.sampling_config.ic = self.geom.sampling_config.domain
                    self.sampling_config.time = self.td.sampling_config.domain
        else:
            check_param_type(sampling_config, "sampling_config", data_type=SamplingConfig)
            self.sampling_config = copy.deepcopy(sampling_config)
            self.geom.set_sampling_config(sampling_config)
            if not sampling_config.time:
                raise ValueError("Undefined sampling info in time, please check your sampling config")
            self.td.set_sampling_config(sampling_config)
            self.td.sampling_config.domain = copy.deepcopy(sampling_config.time)

    def set_sampling_config(self, sampling_config: SamplingConfig):
        """
        set sampling info

        Args:
            sampling_config (SamplingConfig): sampling configuration.

        Raises:
            TypeError: If `sampling_config` is not instance of SamplingConfig.
        """
        if not isinstance(sampling_config, SamplingConfig):
            raise TypeError("sampling_config: {} should be instance of class SamplingConfig, bug got: {}".format(
                sampling_config, type(sampling_config)))
        check_param_type(sampling_config, "sampling_config", data_type=SamplingConfig)

        self.sampling_config = copy.deepcopy(sampling_config)
        self.td.set_sampling_config(self.sampling_config)
        self.td.sampling_config.domain = copy.deepcopy(self.td.sampling_config.time)
        self.geom.set_sampling_config(self.sampling_config)

    def _get_time_samples(self):
        time_points = self.td.sampling(geom_type="domain")
        return time_points

    def _get_geom_domain_samples(self):
        return self.geom.sampling(geom_type="domain")

    def _get_geom_boundary_samples(self):
        return self.geom.sampling(geom_type="BC")

    def _random_merge(self, time_points, geom_points, normal_points=None):
        """random merge"""
        size = len(time_points) - len(geom_points)
        if size > 0:
            rand_index = np.random.randint(low=0, high=len(geom_points), size=size)
            geom_points = np.concatenate([geom_points, geom_points[rand_index]], axis=0)
            if normal_points is not None:
                normal_points = np.concatenate([normal_points, normal_points[rand_index]], axis=0)
        elif size < 0:
            rand_index = np.random.randint(low=0, high=len(time_points), size=-size)
            time_points = np.concatenate([time_points, time_points[rand_index]], axis=0)
        time_geom_points = np.concatenate([geom_points, time_points], axis=-1)
        if normal_points is not None:
            return time_geom_points, normal_points
        return time_geom_points

    def _sampling_domain_samples(self):
        """sample domain data"""
        time_points = self._get_time_samples()
        domain_points = self._get_geom_domain_samples()
        if self.sampling_config.time.random_sampling:
            samples = self._random_merge(time_points, domain_points)
        else:
            length = len(domain_points)
            domain_points = np.repeat(domain_points, len(time_points), axis=0)
            time_points = np.tile(time_points, (length, 1))
            samples = np.concatenate([domain_points, time_points], axis=-1)
        return samples

    def _sampling_boundary_samples(self):
        """sample boundary data"""
        time_points = self._get_time_samples()
        if self.sampling_config.bc.with_normal:
            bc_points, bc_normal = self._get_geom_boundary_samples()
        else:
            bc_points = self._get_geom_boundary_samples()

        if self.sampling_config.time.random_sampling:
            if self.sampling_config.bc.with_normal:
                samples, samples_normal = self._random_merge(time_points, bc_points, normal_points=bc_normal)
            else:
                samples = self._random_merge(time_points, bc_points)
        else:
            length = len(bc_points)
            bd_points = np.repeat(bc_points, len(time_points), axis=0)
            if self.sampling_config.bc.with_normal:
                samples_normal = np.repeat(bc_normal, len(time_points), axis=0)
            time_points = np.tile(time_points, (length, 1))
            samples = np.concatenate([bd_points, time_points], axis=-1)
        if self.sampling_config.bc.with_normal:
            return samples, samples_normal
        return samples

    def sampling(self, geom_type="domain"):
        """
        sampling points

        Args:
            geom_type (str): geometry type: can be ``'domain'`` or ``'BC'`` or ``'IC'``. Default: ``'domain'``.

                - ``'domain'``, feasible domain of the problem.
                - ``'BC'``, boundary of the problem.
                - ``'IC'``, initial condition of the problem.

        Returns:
            Numpy.array, if the with_normal property of boundary configuration is true, returns 2D numpy array with
                         boundary normal vectors. Otherwise, returns 2D numpy array without boundary normal vectors.

        Raises:
            ValueError: If `config` is ``None``.
            KeyError: If `geom_type` is ``'domain'`` but `config.domain` is ``None``.
            KeyError: If `geom_type` is ``'BC'`` but `config.bc` is ``None``.
            KeyError: If `geom_type` is ``'IC'`` but `config.ic` is ``None``.
            ValueError: If `geom_type` is not ``'BC'``, ``'IC'`` nor ``'domain'``.
        """
        config = self.sampling_config
        check_param_type(config, _space.join((self.geom_type, self.name, "'s sampling_config")),
                         data_type=SamplingConfig)
        check_param_type_value(geom_type, _space.join((self.geom_type, self.name, "'s geom_type")),
                               GEOM_TYPES, data_type=str)
        if geom_type.lower() == "domain":
            check_param_type(config.domain, _space.join((self.geom_type, self.name, "'s domain config")),
                             exclude_type=type(None))
            logger.info("Sampling domain points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))

            column_name = self.name + "_domain_points"
            data = self._sampling_domain_samples()
            self.columns_dict["domain"] = [column_name]
            data = data.astype(self.dtype)
            return data
        if geom_type.lower() == "bc":
            check_param_type(config.bc, _space.join((self.geom_type, self.name, "'s bc config")),
                             exclude_type=type(None))
            logger.info("Sampling BC points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))
            if config.bc.with_normal:
                data, data_normal = self._sampling_boundary_samples()
                column_data = self.name + "_BC_points"
                column_normal = self.name + "_BC_normal"
                self.columns_dict["BC"] = [column_data, column_normal]
                data = data.astype(self.dtype)
                data_normal = data_normal.astype(self.dtype)
                return data, data_normal
            data = self._sampling_boundary_samples()
            column_data = self.name + "_BC_points"
            self.columns_dict["BC"] = [column_data]
            data = data.astype(self.dtype)
            return data
        if geom_type.lower() == "ic":
            check_param_type(config.domain, _space.join((self.geom_type, self.name, "'s ic config")),
                             exclude_type=type(None))
            logger.info("Sampling IC points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))
            data = self._sampling_initial_samples()
            column_data = self.name + "_IC_points"
            self.columns_dict["IC"] = [column_data]
            data = data.astype(self.dtype)
            return data
        raise ValueError("Unknown geom_type: {}, only \"domain/BC/IC\" are supported for {}:{}".format(
            geom_type, self.geom_type, self.name))

    def _sampling_initial_samples(self):
        domain_config = self.geom.sampling_config.domain
        self.geom.sampling_config.domain = self.sampling_config.ic
        domain_points = self._get_geom_domain_samples()
        t_start_repeat = np.reshape(np.repeat(self.td.start, len(domain_points), axis=0), (-1, 1))
        xt = np.concatenate([domain_points, t_start_repeat], axis=-1)
        self.geom.sampling_config.domain = domain_config
        return xt
