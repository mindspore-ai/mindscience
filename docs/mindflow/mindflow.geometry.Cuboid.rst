mindflow.geometry.Cuboid
========================

.. py:class:: mindflow.geometry.Cuboid(name, coord_min, coord_max, dtype=<class 'numpy.float32'>, sampling_config=None)

    立方体对象的定义。

    参数：
        - **name** (str) - 立方体对象的名称。
        - **coord_min** (Union[tuple[float, float], tuple[int, int], list[float, float], list[int, int], numpy.ndarray]) - 立方体对象左下角的坐标。
        - **coord_max** (Union[tuple[float, float], tuple[int, int], list[float, float], list[int, int], numpy.ndarray]) - 立方体对象右上角的坐标。
        - **dtype** (numpy.dtype) - 采样点数据类型的数据类型。默认值：numpy.float32。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值：None。
