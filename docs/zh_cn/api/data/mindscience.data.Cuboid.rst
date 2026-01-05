mindscience.data.Cuboid
==========================

.. py:class:: mindscience.data.Cuboid(name, coord_min, coord_max, dtype=numpy.float32, sampling_config=None)

    三维立方体对象的定义。

    参数：
        - **name** (str) - 立方体对象的名称。
        - **coord_min** (Union[tuple[int, int, int], tuple[float, float, float], list[int, int, int], list[float, float, float], numpy.ndarray]) - 立方体对象的左下角坐标。
        - **coord_max** (Union[tuple[int, int, int], tuple[float, float, float], list[int, int, int], list[float, float, float], numpy.ndarray]) - 立方体对象的右上角坐标。
        - **dtype** (numpy.dtype) - 采样点的数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。
