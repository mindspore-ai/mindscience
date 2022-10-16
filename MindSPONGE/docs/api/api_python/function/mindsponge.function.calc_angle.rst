mindsponge.function.calc_angle
==============================

.. py:class:: mindsponge.function.calc_angle(position_a, position_b: Tensor, position_c: Tensor, pbc_box: Tensor = None)

    计算由A，B，C三个位置形成的角。

    参数：
        - **position_a** (Tensor) - 位置a，shape为(B, ..., D)。
        - **position_b** (Tensor) - 位置b，shape为(B, ..., D)。
        - **position_c** (Tensor) - 位置c，shape为(B, ..., D)。
        - **pbc_box** (Tensor) - PBC box，shape为(B, D)。默认值：None。

    输出：
        Tensor。计算所得角。shape为(B, ..., 1)。

    符号：
        - **B** - Batch size。
        - **D** - 模拟系统的维度。