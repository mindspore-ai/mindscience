mindsponge.function.get_kinetic_energy
======================================

.. py:class:: mindsponge.function.get_kinetic_energy(mass, velocity)

    获取计算模拟系统的动能。

    参数：
        - **mass** (Tensor) - 系统中原子的质量。
        - **velocity** (Tensor) - 系统中原子的速度。

    输出：
        Tensor。动能。

    符号：
        - **B** - Batch size。
        - **A** - 模拟系统中原子总数。
        - **D** - 模拟系统的维度。