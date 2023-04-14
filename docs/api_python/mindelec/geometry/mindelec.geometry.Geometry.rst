mindelec.geometry.Geometry
==========================

.. py:class:: mindelec.geometry.Geometry(name, dim, coord_min, coord_max, dtype=np.float32, sampling_config=None)

    几何对象的定义。

    参数：
        - **name** (str) - 几何体的名称。
        - **dim** (int) - 维数。
        - **coord_min** (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]) - 几何体的最小坐标。
        - **coord_max** (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]) - 几何体的最大坐标。
        - **dtype** (numpy.dtype) - 采样点数据类型的数据类型。默认值： ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。

    .. py:method:: mindelec.geometry.Geometry.set_name(name)

        设置几何实例名称。

        参数：
            - **name** (str) - 几何实例的名称。

        异常：
            - **TypeError** - 如果 `name` 不是字符串。

    .. py:method:: mindelec.geometry.Geometry.set_sampling_config(sampling_config: SamplingConfig)

        设置采样信息。

        参数：
            - **sampling_config** (SamplingConfig) - 采样配置。

        异常：
            - **TypeError** - 如果 `sampling_config` 不是SamplingConfig的实例。