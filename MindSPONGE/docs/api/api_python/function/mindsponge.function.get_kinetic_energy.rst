mindsponge.function.get_kinetic_energy
======================================

.. py:class:: mindsponge.function.get_kinetic_energy(mass, velocity)

    获取计算模拟系统的动能。

    参数：
        - **mass** (Tensor) - 系统中原子的质量，shape为(B, A)。
        - **velocity** (Tensor) - 系统中原子的速度，shape为(B, A, D)。

    输出：
        Tensor。动能，shape为(B)。

    符号：
        - **B** - Batch size。
        - **A** - 模拟系统中原子总数。
        - **D** - 模拟系统的维度。