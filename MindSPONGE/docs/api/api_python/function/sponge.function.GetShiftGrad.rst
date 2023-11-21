sponge.function.GetShiftGrad
================================

.. py:class:: sponge.function.GetShiftGrad(num_atoms: int, bonds: Tensor, num_walkers: int = 1, dimension: int = 3, use_pbc: bool = None)

    计算维度为 :math:`(K, N, D)` 的B矩阵的微分，N是原子的邻居原子数，D是模拟系统的维度。

    参数：
        - **num_atoms** (int) - 系统中原子总数。
        - **bonds** (Tensor) - 需要被约束的键，shape为 :math:`(C, 2)` ，数据类型是int。
        - **num_walkers** (int) - 多线程的数量。默认值： 1。
        - **dimension** (int) - 维度数量。默认值： 3。
        - **use_pbc** (bool) - 是否使用周期性边界条件。如果是 ``None`` ，则根据是否提供pbc_box决定是否在周期性边界中计算距离。默认值： ``None`` 。

    .. py:method:: construct(coordinate_new: Tensor, coordinate_old: Tensor, pbc_box: Tensor = None)

        用来计算 B 矩阵差异的模块，其维度为：K*N*D。

        参数：
            - **coordinate_new** (Tensor) - 系统的新坐标。张量的shape为 :math:`(B, A, D)` 。数据类型为float。
            - **coordinate_old** (Tensor) - 系统的旧坐标。张量的shape为 :math:`(B, A, D)` 。数据类型为float。
            - **pbc_box** (Tensor) - PBC box 的张量。张量的shape为 :math:`(B, D)` 。数据类型为float。

        返回：
            Tensor。shift。张量的shape为 :math:`(B, A, D)` 。数据类型为float。
