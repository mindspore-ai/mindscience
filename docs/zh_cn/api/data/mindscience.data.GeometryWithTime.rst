mindscience.data.GeometryWithTime
===================================

.. py:class:: mindscience.data.GeometryWithTime(geometry, timedomain, sampling_config=None)

    带时间维度的几何对象定义。

    参数：
        - **geometry** (Geometry) - 空间几何对象。
        - **timedomain** (TimeDomain) - 时间域对象。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。

    异常：
        - **ValueError** - 当 `sampling_config` 不为 ``None`` 但 `sampling_config.time` 为 ``None`` 时抛出。

    .. py:method:: sampling(geom_type="domain")

        采样点。

        参数：
            - **geom_type** (str) - 几何类型：支持 ``'domain'``、``'BC'`` 或 ``'IC'``，默认 ``'domain'``。

              - ``'domain'``：问题的可行域（feasible domain of the problem）。
              - ``'BC'``：问题的边界（boundary of the problem）。
              - ``'IC'``：问题的初始条件（initial condition of the problem）。

        返回：
            Numpy.array。若边界配置 `with_normal` 为 True，则返回 **两项** （采样点二维数组与对应法线二维数组）；否则返回 **一项** （采样点二维数组）。

        异常：
            - **KeyError** - 当 `geom_type` 为 ``'domain'`` 但 ``self.sampling_config.domain`` 为 ``None`` 时抛出。
            - **KeyError** - 当 `geom_type` 为 ``'BC'`` 但 ``self.sampling_config.bc`` 为 ``None`` 时抛出。
            - **KeyError** - 当 `geom_type` 为 ``'IC'`` 但 ``self.sampling_config.ic`` 为 ``None`` 时抛出。

    .. py:method:: set_sampling_config(sampling_config)

        设置采样信息。

        参数：
            - **sampling_config** (SamplingConfig) - 采样配置。

        异常：
            - **TypeError** - 当 `sampling_config` 不是 `SamplingConfig` 的实例时抛出。 
