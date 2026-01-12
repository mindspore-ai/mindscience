mindscience.data.Disk
==========================

.. py:class:: mindscience.data.Disk(name, center, radius, dtype=numpy.float32, sampling_config=None)

    二维圆盘区域对象的定义。

    参数：
        - **name** (str) - 圆盘名称。
        - **center** (Union[tuple, list, numpy.ndarray]) - 圆心坐标。支持单值、list、tuple或数组。当参数类型为 tuple 或 list 时，其元素应为 int 或 float 类型，且其长度必须为 ``2``。
        - **radius** (Union[int, float]) - 圆盘半径。
        - **dtype** (numpy.dtype) - 采样点数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。

    异常：
        - **ValueError** - 当 `center` 不是长度为2的list或长度为2的tuple时。
        - **ValueError** - 当 `radius` 为负数时。

    .. py:method:: sampling(geom_type="domain")

        采样区域或边界点。

        参数：
            - **geom_type** (str) - 几何类型，可选 ``'domain'`` （问题可行域）或 ``'BC'`` （边界），默认 ``'domain'``。

              - ``'domain'``：问题的可行域（feasible domain of the problem）。
              - ``'BC'``：问题的边界（boundary of the problem）。

        返回：
            Numpy.array。如果配置选择包括法向向量（ `with_normal=True` ），返回带边界法向向量的二维numpy数组。否则返回不带边界法向向量的二维numpy数组。

        异常：
            - **KeyError** - 如果 `geom_type` 是 ``'domain'``，但 ``self.sampling_config.domain`` 是 ``None``。
            - **KeyError** - 如果 `geom_type` 为 ``'BC'``，但 ``self.sampling_config.bc`` 为 ``None``。
            - **ValueError** - 如果 `geom_type` 既不是 ``'BC'`` 也不是 ``'domain'``。


