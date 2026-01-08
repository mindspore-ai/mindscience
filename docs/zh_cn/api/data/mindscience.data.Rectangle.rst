mindscience.data.Rectangle
=============================

.. py:class:: mindscience.data.Rectangle(name, coord_min, coord_max, dtype=numpy.float32, sampling_config=None)

    二维矩形区域的定义。

    参数：
        - **name** (str) - 矩形名称。
        - **coord_min** (Union[tuple, list, numpy.ndarray]) - 矩形左下角坐标。当参数类型为 tuple 或 list 时，其元素应为 int 或 float 类型，且其长度必须为 ``2``。
        - **coord_max** (Union[tuple, list, numpy.ndarray]) - 矩形右上角坐标。当参数类型为 tuple 或 list 时，其元素应为 int 或 float 类型，且其长度必须为 ``2``。
        - **dtype** (numpy.dtype) - 采样点的数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。
