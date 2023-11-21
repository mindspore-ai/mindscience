sponge.function.GetDistanceShift
====================================

.. py:class:: sponge.function.GetDistanceShift(bonds: Tensor, num_atoms: int, num_walkers: int = 1, use_pbc: bool = None)

    计算维数为C的B矩阵。

    参数：
        - **bonds** (Tensor) - 需要被约束的键，shape为 :math:`(C, 2)` 。
        - **num_atoms** (int) - 系统中原子总数。
        - **num_walkers** (int) - 多线程的数量。默认值：1。
        - **use_pbc** (bool) - 是否使用周期性边界条件。默认值： ``None``。
    
    .. py:method:: construct(coordinate_new: Tensor, coordinate_old: Tensor, pbc_box: Tensor = None)

        用于计算 B 矩阵的模块，其维度为：C。

        参数：
            - **coordinate_new** (Tensor) - 系统的新坐标。张量的shape为 :math:`(B, A, D)` 。数据类型为float。
            - **coordinate_old** (Tensor) - 系统的旧坐标。张量的shape为 :math:`(B, A, D)` 。数据类型为float。
            - **pbc_box** (Tensor) - PBC box 的张量。张量的shape为 :math:`(B, D)` 。数据类型为float。

        返回：
            Tensor。shift。张量的shape为 :math:`(B, A, D)` 。数据类型为float。
