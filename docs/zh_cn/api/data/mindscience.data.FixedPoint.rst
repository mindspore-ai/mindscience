mindscience.data.FixedPoint
==============================

.. py:class:: mindscience.data.FixedPoint(name, coord, dtype=numpy.float32, sampling_config=None)

    一个固定点几何对象的定义。

    参数：
        - **name** (str) - 定义几何体的名称。
        - **coord** (Union[int, float, tuple, list, numpy.ndarray]) - 固定点坐标。当参数类型为 tuple 或 list 时，其元素应为 int 或 float 类型。
        - **dtype** (numpy.dtype) - 采样点数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。

    .. py:method:: sampling(geom_type="domain")

        采样点。

        参数：
            - **geom_type** (str) - 几何类型，支持 ``'domain'`` 与 ``'BC'``，默认 ``'domain'``。

        返回：
            Numpy.ndarray。若边界配置启用了法线则返回带法线向量的二维数组，否则返回普通二维数组。

        异常：
            - **KeyError** - 当 `geom_type` 是 ``'domain'`` 但 ``self.sampling_config.domain`` 为 ``None`` 时抛出。
            - **KeyError** - 当 `geom_type` 是 ``'BC'`` 但 ``self.sampling_config.bc`` 为 ``None`` 时抛出。
            - **ValueError** - 当 `geom_type` 既不是 ``'BC'`` 也不是 ``'domain'`` 时抛出。
 

