mindscience.data.Rectangle
=============================

.. py:class:: mindscience.data.Rectangle(name, coord_min, coord_max, dtype=numpy.float32, sampling_config=None)

    二维矩形区域的定义。

    参数：
        - **name** (str) - 矩形名称。
        - **coord_min** (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]) - 矩形左下角坐标。
        - **coord_max** (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]) - 矩形右上角坐标。
        - **dtype** (numpy.dtype) - 采样点的数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。
