mindelec.geometry.Rectangle
===========================

.. py:class:: mindelec.geometry.Rectangle(name, coord_min, coord_max, dtype=np.float32, sampling_config=None)

    矩形对象的定义。

    参数：
        - **name** (str) - 矩形的名称。
        - **coord_min** (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]) - 矩形的左底部的坐标。
        - **coord_max** (Union[tuple[int, int], tuple[float, float], list[int, int], list[float, float], numpy.ndarray]) - 矩形的右顶部的坐标。
        - **dtype** (numpy.dtype) - 采样点数据类型的数据类型。默认值： ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。
