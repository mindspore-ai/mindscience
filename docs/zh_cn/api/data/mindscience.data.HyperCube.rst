mindscience.data.HyperCube
=============================

.. py:class:: mindscience.data.HyperCube(name, dim, coord_min, coord_max, dtype=numpy.float32, sampling_config=None)

    超立方体（多维长方体）几何对象的定义。

    参数：
        - **name** (str) - 超立方体名称。
        - **dim** (int) - 维度数量。
        - **coord_min** (Union[int, float, tuple, list, numpy.ndarray]) - 超立方体的最小坐标。若类型为 tuple/list，其元素可为 ``tuple[int, int]``、``tuple[float, float]``、``list[int, int]``、``list[float, float]`` 等形式。
        - **coord_max** (Union[int, float, tuple, list, numpy.ndarray]) - 超立方体的最大坐标。若类型为 tuple/list，其元素可为 ``tuple[int, int]``、``tuple[float, float]``、``list[int, int]``、``list[float, float]`` 等形式。
        - **dtype** (numpy.dtype) - 采样点数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。

    异常：
        - **TypeError** - 当 `sampling_config` 不是 `SamplingConfig` 类实例时抛出。

    .. py:method:: sampling(geom_type="domain")

        采样点。

        参数：
            - **geom_type** (str) - 几何类型：可为 ``'domain'`` 或 ``'BC'``，默认 ``'domain'``。
            
              - ``'domain'``：问题的可行域（feasible domain of the problem）。
              - ``'BC'``：问题的边界（boundary of the problem）。

        返回：
            Numpy.ndarray。若边界配置 `with_normal` 为 True，则返回带边界法线向量的二维数组；否则返回不带法线向量的二维数组。

        异常：
            - **KeyError** - 当 `geom_type` 为 ``'domain'`` 但 ``self.sampling_config.domain`` 为 ``None`` 时抛出。
            - **KeyError** - 当 `geom_type` 为 ``'BC'`` 但 ``self.sampling_config.bc`` 为 ``None`` 时抛出。
            - **ValueError** - 当 `geom_type` 既不是 ``'BC'`` 也不是 ``'domain'`` 时抛出。
