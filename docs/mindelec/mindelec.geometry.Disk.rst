mindelec.geometry.Disk
======================

.. py:class:: mindelec.geometry.Disk(name, center, radius, dtype=<class 'numpy.float32'>, sampling_config=None)

    圆盘对象的定义。

    参数：
        - **name** (str) - 圆盘的名称。
        - **center** (Union[tuple[int, float], list[int, float], numpy.ndarray]) - 圆盘的中心坐标。
        - **radius** (Union[int, float]) - 圆盘的半径。
        - **dtype** (numpy.dtype) - 采样点的数据类型。默认值：numpy.float32。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值：None。

    异常：
        - **ValueError** - 如果 `center` 既不是长度为2的list也不是Tuple。
        - **ValueError** - 如果 `radius` 为负数。

    .. py:method:: mindelec.geometry.Disk.sampling(geom_type='domain')

        采样域和边界点。

        参数：
            - **geom_type** (str) - 几何类型。

        返回：
            Numpy.array，带或不带边界法向向量的二维numpy数组。

        异常：
            - **ValueError** - 如果 `config` 为None。
            - **KeyError** - 如果 `geom_type` 是 `domain`，但 `config.domain` 是None。
            - **KeyError** - 如果 `geom_type `为 `BC`，但 `config.bc` 为None。
            - **ValueError** - 如果 `geom_type` 既不是 `BC` 也不是 `domain`。

