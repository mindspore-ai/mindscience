mindscience.data.Interval
============================

.. py:class:: mindscience.data.Interval(name, coord_min, coord_max, dtype=numpy.float32, sampling_config=None)

    一维区间对象的定义。

    参数：
        - **name** (str) - 区间名称。
        - **coord_min** (Union[int, float]) - 区间左边界。
        - **coord_max** (Union[int, float]) - 区间右边界。
        - **dtype** (numpy.dtype) - 采样点的数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。

    异常：
        - **ValueError** - 当 `coord_min` 或 `coord_max` 不是 int/float 时。
