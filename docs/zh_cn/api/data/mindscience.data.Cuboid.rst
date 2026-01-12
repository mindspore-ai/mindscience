mindscience.data.Cuboid
==========================

.. py:class:: mindscience.data.Cuboid(name, coord_min, coord_max, dtype=numpy.float32, sampling_config=None)

    三维立方体对象的定义。

    参数：
        - **name** (str) - 立方体对象的名称。
        - **coord_min** (Union[tuple, list, numpy.ndarray]) - 立方体对象的左下角坐标。当参数类型为 tuple 或 list 时，其元素应为 int 或 float 类型，且其长度必须为 ``3``。
        - **coord_max** (Union[tuple, list, numpy.ndarray]) - 立方体对象的右上角坐标。当参数类型为 tuple 或 list 时，其元素应为 int 或 float 类型，且其长度必须为 ``3``。
        - **dtype** (numpy.dtype) - 采样点的数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。
