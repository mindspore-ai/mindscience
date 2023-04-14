mindflow.geometry.Interval
==========================

.. py:class:: mindflow.geometry.Interval(name, coord_min, coord_max, dtype=np.float32, sampling_config=None)

    区间对象的定义。

    参数：
        - **name** (str) - 区间的名称。
        - **coord_min** (Union[int, float]) - 区间左边界。
        - **coord_max** (Union[int, float]) - 区间右边界。
        - **dtype** (numpy.dtype) - 采样点数据类型的数据类型。默认值： ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。

    异常：
        - **ValueError** - 如果 `coord_min` 或 `coord_max` 既不是int也不是float。
