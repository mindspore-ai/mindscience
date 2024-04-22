sponge.metrics.Metric
============================

.. py:class:: sponge.metrics.Metric()

    Metric是用于评估模拟系统状态和性能的基本工具。它提供了一种机制来跟踪模拟系统中各种物理量的变化。Metric的基类定义了一组方法，用于更新模拟系统的状态信息并计算相应的指标。

    .. py:method:: update(coordinate: Tensor, pbc_box: Tensor = None, energy: Tensor = None, force: Tensor = None, potentials: Tensor = None, total_bias: Tensor = None, biases: Tensor = None)

        更新模拟系统的状态信息。

        参数：
            - **coordinate** (Tensor) - shape为 :math:`(B, A, D)` 的Tensor。数据类型为 float。系统中原子的位置坐标。
            - **pbc_box** (Tensor，可选) - shape为 :math:`(B, D)` 的Tensor。数据类型为 float。PBC box的Tensor。默认值： ``None``。
            - **energy** (Tensor，可选) - shape为 :math:`(B, 1)` 的Tensor。数据类型为 float。模拟系统的总能量。默认值： ``None``。
            - **force** (Tensor，可选) - shape为 :math:`(B, A, D)` 的Tensor。数据类型为 float。模拟系统中每个原子的受力。默认值： ``None``。
            - **potentials** (Tensor，可选) - shape为 :math:`(B, U)` 的Tensor。数据类型为 float。所有势能。默认值： ``None``。
            - **total_bias** (Tensor，可选) - shape为 :math:`(B, 1)` 的Tensor。数据类型为 float。所有总偏差势能。默认值： ``None``。
            - **biases** (Tensor，可选) - shape为 :math:`(B, V)` 的Tensor。数据类型为 float。所有偏差势能。默认值： ``None``。

        .. note::
            - B: 代表模拟中的walkers数量。
            - A: 代表模拟系统的原子数量。
            - D: 代表模拟系统的空间维度。通常为3。
            - U: 代表势能的数量。
            - V: 代表偏置势能的数量。
