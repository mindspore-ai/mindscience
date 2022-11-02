mindsponge.function.calc_angle_with_pbc
=======================================

.. py:function:: mindsponge.function.calc_angle_with_pbc(position_a, position_b, position_c, pbc_box)

    在有周期性边界条件的情况下计算由A，B，C三个位置形成的角。

    参数：
        - **position_a** (Tensor) - 位置a，shape为(B, ..., D)。
        - **position_b** (Tensor) - 位置b，shape为(B, ..., D)。
        - **position_c** (Tensor) - 位置c，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，shape为(B, D)。

    输出：
        Tensor。计算所得角。shape为(B, ..., 1)。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度。