mindflow.geometry.Cylinder
==========================

.. py:class:: mindflow.geometry.Cylinder(name, centre, radius, h_min, h_max, h_axis, boundary_type='uniform', dtype=numpy.float32, sampling_config=None)

    圆柱体对象的定义。

    参数：
        - **name** (str) - 圆柱体的名称。
        - **centre** (numpy.ndarray) - 底部的原点。
        - **radius** (float) - 底部的半径。
        - **h_min** (float) - 底部的高度坐标。
        - **h_max** (float) - 顶部的高度坐标。
        - **h_axis** (int) - 底部法向向量的轴。
        - **boundary_type** (str) - 值可以是 ``'uniform'`` 或 ``'unweighted'`` 。默认值： ``'uniform'``。

          - ``'uniform'``，每个边界中的预期样本数与边界的面积（长度）是成比例的。
          - ``'unweighted'``，每个边界中的预期样本数相同。

        - **dtype** (numpy.dtype) - 采样点数据类型的数据类型。默认值： ``np.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。
