mindelec.geometry.HyperCube
===========================

.. py:class:: mindelec.geometry.HyperCube(name, dim, coord_min, coord_max, dtype=np.float32, sampling_config=None)

    超立方体对象的定义。

    参数：
        - **name** (str) - 超立方体的名称。
        - **dim** (int) - 维数。
        - **coord_min** (Union[int, float, tuple, list, numpy.ndarray]) - 超立方体的最小坐标。若参数类型为tuple或list，元素类型支持tuple[int, int]，tuple[float, float]，list[int, int]，list[float, float]。
        - **coord_max** (Union[int, float, tuple, list, numpy.ndarray]) - 超立方体的最大坐标。若参数类型为tuple或list，元素类型支持tuple[int, int]，tuple[float, float]，list[int, int]，list[float, float]。
        - **dtype** (numpy.dtype) - 采样点数据类型的数据类型。默认值： ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。

    异常：
        - **TypeError** - `sampling_config` 不是类采样配置的实例。


    .. py:method:: mindelec.geometry.HyperCube.sampling(geom_type='domain')

        采样点。

        参数：
            - **geom_type** (str) - 几何类型，可以是 ``"domain"`` 或者 ``"BC"`` 。默认值： ``"domain"``。

              - ``"domain"``: 问题的可行域。
              - ``"BC"``: 问题的边界条件。

        返回：
            Numpy.array，如果配置选择包括法向向量，返回带边界法向向量的二维numpy数组。否则返回不带边界法向向量的二维numpy数组。

        异常：
            - **ValueError** - 如果 `config` 为 ``None``。
            - **KeyError** - 如果 `geom_type` 为 ``"domain"``，但 `config.domain` 为 ``None``。
            - **KeyError** - 如果 `geom_type` 为 ``"BC"``，但 `config.bc` 为 ``None``。
            - **ValueError** - 如果 `geom_type` 既不是 ``"BC"`` 也不是 ``"domain"``。
