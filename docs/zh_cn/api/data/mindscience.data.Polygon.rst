mindscience.data.Polygon
===========================

.. py:class:: mindscience.data.Polygon(name, vertices, boundary_type="uniform", dtype=numpy.float32, sampling_config=None)

    通用多边形对象的定义。

    参数：
        - **name** (str) - 多边形对象的名称。
        - **vertices** (numpy.ndarray) - 多边形顶点数组（按逆时针顺序）。
        - **boundary_type** (str) - 边界采样策略，默认 ``'uniform'``。
 
          - ``'uniform'``：按照各边界的“面积/长度”占比分配采样点。
          - ``'unweighted'``：对所有边/面平均分配采样点数量。

        - **dtype** (numpy.dtype) - 采样点的数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。
