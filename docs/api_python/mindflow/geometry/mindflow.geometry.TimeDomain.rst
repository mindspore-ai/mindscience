mindflow.geometry.TimeDomain
============================

.. py:class:: mindflow.geometry.TimeDomain(name, start=0.0, end=1.0, dtype=np.float32, sampling_config=None)

    时域的定义。

    参数：
        - **name** (str) - 时域名称。
        - **start** (Union[int, float]) - 时域的开始。默认值： ``0.0``。
        - **end** (Union[int, float]) - 时域结束。默认值： ``1.0``。
        - **dtype** (numpy.dtype) - 采样点数据类型的数据类型。默认值： ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。
