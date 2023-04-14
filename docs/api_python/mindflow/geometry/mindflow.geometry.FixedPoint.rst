mindflow.geometry.FixedPoint
============================

.. py:class:: mindflow.geometry.FixedPoint(name, coord, dtype=np.float32, sampling_config=None)

    固定点对象的定义。

    参数：
        - **name** (str) - 定点的名称。
        - **coord** (Union[int, float, tuple, list, numpy.ndarray]) - 定点坐标。若参数类型为tuple或list，元素类型支持tuple[int, int]，tuple[float, float]，list[int, int]，list[float, float]。
        - **dtype** (numpy.dtype) - 采样点数据类型的数据类型。默认值： ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。

    .. py:method:: sampling(geom_type='domain')

        采样点。

        参数：
            - **geom_type** (str) - 几何类型，支持 ``'domain'`` 和 ``'BC'``。默认值： ``'domain'``。

        返回：
            Numpy.ndarray，带或不带边界法向向量的二维numpy数组。

        异常：
            - **ValueError** - 如果 `config` 为 ``None``。
            - **KeyError** - 如果 `geom_type` 是 ``'domain'`` ，但 `config.domain` 为 ``None``。
            - **KeyError** - 如果 `geom_type` 为 ``'BC'`` ，但 `config.bc` 为 ``None``。
            - **ValueError** - 如果 `geom_type` 既不是 ``'BC'`` 也不是 ``'domain'`` 。