sponge.function.GetDistanceShift
====================================

.. py:class:: sponge.function.GetDistanceShift(bonds: Tensor, num_atoms: int, num_walkers: int = 1, use_pbc: bool = None)

    计算维数为C的B矩阵。

    参数：
        - **bonds** (Tensor) - 需要被约束的键，shape为(C, 2)。
        - **num_atoms** (int) - 系统中原子总数。
        - **num_walkers** (int) - 多线程的数量。默认值：1。
        - **use_pbc** (bool) - 是否使用周期性边界条件。默认值："None"。

    输出：
        Tensor。计算所得转换B矩阵，shape为(B, A, D)。

    符号：
        - **B** - Batch size。
        - **A** - 模拟系统中原子总数。
        - **D** - 模拟系统的维度。