sponge.function.GetShiftGrad
================================

.. py:class:: sponge.function.GetShiftGrad(num_atoms: int, bonds: Tensor, num_walkers: int = 1, dimension: int = 3, use_pbc: bool = None)

    计算维度为(K, N, D)的B矩阵的微分。

    参数：
        - **bonds** (Tensor) - 需要被约束的键，shape为(K, N, D)。
        - **num_atoms** (int) - 系统中原子总数。
        - **num_walkers** (int) - 多线程的数量。默认值：1。
        - **dimension** (int) - 维度数量。默认值：3。
        - **use_pbc** (bool) - 是否使用周期性边界条件。默认值："None"。

    输出：
        Tensor。计算所得B矩阵的微分，shape为(B, A, D)。

    符号：
        - **B** - Batch size。
        - **A** - 模拟系统中原子总数。
        - **N** - 原子的邻居原子数。
        - **D** - 模拟系统的维度。