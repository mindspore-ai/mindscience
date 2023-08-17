sponge.function.calc_torsion_pbc
=========================================

.. py:function:: sponge.function.calc_torsion_pbc(position_a, position_b, position_c, position_d, pbc_box)

    在有周期性边界条件的情况下计算由四个位置A，B，C，D形成的扭转角。

    参数：
        - **position_a** (Tensor) - 位置a，shape为(B, ..., D)。
        - **position_b** (Tensor) - 位置b，shape为(B, ..., D)。
        - **position_c** (Tensor) - 位置c，shape为(B, ..., D)。
        - **position_d** (Tensor) - 位置d，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，shape为(B, D)。

    输出：
        Tensor。计算所得扭转角。shape为(B, ..., 1)。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度。