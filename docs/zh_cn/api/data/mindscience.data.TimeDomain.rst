mindscience.data.TimeDomain
==============================

.. py:class:: mindscience.data.TimeDomain(name, start=0.0, end=1.0, dtype=numpy.float32, sampling_config=None)

    时间域的定义。

    参数：
        - **name** (str) - 时域名称。
        - **start** (Union[int, float]) - 时域起点，默认 ``0.0``。
        - **end** (Union[int, float]) - 时域终点，默认 ``1.0``。
        - **dtype** (numpy.dtype) - 采样点数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。
