sponge.metrics.MetricCV
============================

.. py:class:: sponge.metrics.MetricCV(colvar: Colvar)

    集体变量（CVs）的度量标准。

    .. py:method:: get_unit(units: Units = None)

        返回集体变量的单位。

        参数：
            - **units** (Units，可选) - 用于指定集体变量单位的对象。默认值： ``None``。

    .. py:method:: update(coordinate: Tensor, pbc_box: Tensor = None, energy: Tensor = None, force: Tensor = None, potentials: Tensor = None, total_bias: Tensor = None, biases: Tensor = None)

        更新系统的状态信息。

        参数：
            - **coordinate** (Tensor) - shape为 :math:`(B, A, D)` 的Tensor。数据类型为 float。原子在系统中的位置坐标。
            - **pbc_box** (Tensor，可选) - shape为 :math:`(B, D)` 的Tensor。数据类型为 float。PBC 盒的Tensor。默认值： ``None``。
            - **energy** (Tensor，可选) - shape为 :math:`(B, 1)` 的Tensor。数据类型为 float。模拟系统的总势能。默认值： ``None``。
            - **force** (Tensor，可选) - shape为 :math:`(B, A, D)` 的Tensor。数据类型为 float。模拟系统中每个原子的力。默认值： ``None``。
            - **potentials** (Tensor，可选) - shape为 :math:`(B, U)` 的Tensor。数据类型为 float。来自力场的原始势能。默认值： ``None``。
            - **total_bias** (Tensor，可选) - shape为 :math:`(B, 1)` 的Tensor。数据类型为 float。用于重加权的总偏差能量。默认值： ``None``。
            - **biases** (Tensor，可选) - shape为 :math:`(B, V)` 的Tensor。数据类型为 float。来自偏差函数的原始偏差势能。默认值： ``None``。

        .. note:: 
            - B: 批量大小，例如模拟中的walkers数量。
            - A: 模拟系统中的原子数。
            - D: 模拟系统空间的维数。通常为 3。
            - U: 势能的数量。
            - V: 偏差势能的数量。


