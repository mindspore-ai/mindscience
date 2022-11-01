mindsponge.function.calc_angle_without_pbc
==========================================

.. py:function:: mindsponge.function.calc_angle_without_pbc(position_a, position_b, position_c)

    在没有周期性边界条件的情况下计算由A，B，C三个位置形成的角。

    参数：
        - **position_a** (Tensor) - 位置a，shape为(..., D)。
        - **position_b** (Tensor) - 位置b，shape为(..., D)。
        - **position_c** (Tensor) - 位置c，shape为(..., D)。

    输出：
        Tensor。计算所得角。shape为(..., 1)。

    符号：
        - **D** - 模拟系统的维度。