mindsponge.potential.Bias
=========================

.. py:class:: mindsponge.potential.Bias(colvar, multiple_walkers=False, length_unit, energy_unit, units, use_pbc)

    偏置势场的基础层。

    参数：
        - **colvar** (Colvar) - 可收集变量。
        - **multiple_walkers** (bool) - 是否使用多线程。
        - **length_unit** (str) - 长度单位。
        - **energy_unit** (str) - 能量单位。
        - **units** (Units) - 长度和能量单位。
        - **use_pbc** (bool) - 是否使用PBC。
    
    输出：
        Tensor。势能，shape为(B, 1)。

    符号：
        - **B** - Batch size。
        - **A** - 原子数量。
        - **D** - 模拟系统的维度。

    .. py:method:: update(coordinates, pbc_box=None)

        更新偏置势场的参数。

        参数：
            - **coordinates** (Tensor) - 系统中原子的位置坐标，shape为(B, A, D)。
            - **pbc_box** (Tensor, 可选) - PBC box，shape为(B, D)或(1, D)。默认值："None"。