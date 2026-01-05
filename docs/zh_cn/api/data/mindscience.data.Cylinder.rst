mindscience.data.Cylinder
============================

.. py:class:: mindscience.data.Cylinder(name, centre, radius, h_min, h_max, h_axis, boundary_type="uniform", dtype=numpy.float32, sampling_config=None)

    圆柱体对象的定义。

    参数：
        - **name** (str) - 圆柱体对象的名称。
        - **centre** (numpy.ndarray) - 圆柱体底面的圆心坐标。
        - **radius** (float) - 圆柱体截面半径。
        - **h_min** (float) - 圆柱体底面的高度坐标。
        - **h_max** (float) - 圆柱体顶面的高度坐标。
        - **h_axis** (int) - 圆柱体底面法向量所对应的坐标轴索引。
        - **boundary_type** (str) - 边界采样策略，默认 ``'uniform'``。
 
          - ``'uniform'``：按照各边界的“面积/长度”占比分配采样点。
          - ``'unweighted'``：对所有边/面平均分配采样点数量。

        - **dtype** (numpy.dtype) - 采样点的数据类型，默认 ``numpy.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置，默认 ``None``。
