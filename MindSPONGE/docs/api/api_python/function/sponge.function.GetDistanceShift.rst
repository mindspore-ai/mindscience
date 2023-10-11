sponge.function.GetDistanceShift
====================================

.. py:class:: sponge.function.GetDistanceShift(bonds: Tensor, num_atoms: int, num_walkers: int = 1, use_pbc: bool = None)

    计算维数为C的B矩阵。

    参数：
        - **bonds** (Tensor) - 需要被约束的键，shape为(C, 2)。
        - **num_atoms** (int) - 系统中原子总数。
        - **num_walkers** (int) - 多线程的数量。默认值：1。
        - **use_pbc** (bool) - 是否使用周期性边界条件。默认值："None"。
