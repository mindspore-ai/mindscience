mindelec.geometry.GeometryWithTime
==================================

.. py:class:: mindelec.geometry.GeometryWithTime(geometry, timedomain, sampling_config=None)

    含时域几何体的定义。

    参数：
        - **geometry** (Geometry) - 几何体。
        - **timedomain** (TimeDomain) - 时域。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。

    异常：
        - **ValueError** - 如果 `sampling_config` 不是 ``None``，但 `sampling_config.time` 是 ``None``。

    .. py:method:: mindelec.geometry.GeometryWithTime.sampling(geom_type='domain')

        采样点。

        参数：
            - **geom_type** (str) - 几何类型，可以是 ``"domain"`` 或者 ``"BC"`` 或者 ``"IC"``。默认值： ``"domain"``。

              - ``"domain"``：问题的可行域。
              - ``"BC"``：问题的边界条件。
              - ``"IC"``：问题的初始条件。

        返回：
            Numpy.array，如果在边界条件 ``"BC"`` 中的法向向量 `with_normal` 设置为真，返回带边界法向向量的二维numpy数组。否则返回不带边界法向向量的二维numpy数组。

        异常：
            - **ValueError** - 如果 `config` 为 ``None``。
            - **KeyError** - 如果 `geom_type` 是 ``"domain"``，但 `config.domain` 为 ``None``。
            - **KeyError** - 如果 `geom_type` 为 ``"BC"``，但 `config.bc` 为 ``None``。
            - **KeyError** - 如果 `geom_type` 为 ``"IC"``，但 `config.ic` 为 ``None``。
            - **ValueError** - 如果 `geom_type` 不是 ``"BC"`` 、 ``"IC"`` 也不是 ``"domain"``。

    .. py:method:: mindelec.geometry.GeometryWithTime.set_sampling_config(sampling_config: SamplingConfig)

        设置采样信息。

        参数：
            - **sampling_config** (SamplingConfig) - 采样配置。

        异常：
            - **TypeError** - 如果 `sampling_config` 不是SamplingConfig的实例。
