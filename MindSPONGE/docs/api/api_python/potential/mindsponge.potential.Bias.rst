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
        Tensor。势能。

    .. py:method:: update(coordinates, pbc_box)

        更新偏置势场的参数。

        参数：
            - **coordinates** (Tensor) - 系统中原子的位置坐标。
            - **pbc_box** (Tensor, 可选) - PBC box。