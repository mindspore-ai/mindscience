mindflow.geometry.Tetrahedron
=============================

.. py:class:: mindflow.geometry.Tetrahedron(name, vertices, boundary_type='uniform', dtype=numpy.float32, sampling_config=None)

    四面体对象的定义。

    参数：
        - **name** (str) - 四面体的名称。
        - **vertices** (numpy.ndarray) - 四面体的顶点。
        - **boundary_type** (str) - 值可以是 ``'uniform'`` 或 ``'unweighted'``。默认值： ``'uniform'``。

          - ``'uniform'``，每个边界中的预期样本数与边界的面积（长度）成比例的。
          - ``'unweighted'``，每个边界中的预期样本数相同。

        - **dtype** (numpy.dtype) - 采样点数据类型的数据类型。默认值： ``np.float32``。
        - **sampling_config** (SamplingConfig) - 采样配置。默认值： ``None``。
