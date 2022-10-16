mindsponge.function.GetDistanceShift
====================================

.. py:class:: mindsponge.function.GetDistanceShift(bonds, num_atoms, num_walkers=1, use_pbc)

    计算维数为C的B矩阵。

    参数：
        - **bonds** (Tensor) - 需要被约束的键。
        - **num_atoms** (int) - 系统中原子总数。
        - **num_walkers** (int) - 多线程的数量。默认值：1。
        - **use_pbc** (bool) - 是否使用周期性边界条件。

    输出：
        Tensor。计算所得转换B矩阵。