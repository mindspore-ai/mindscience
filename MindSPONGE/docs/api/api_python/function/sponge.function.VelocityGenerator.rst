sponge.function.VelocityGenerator
=====================================

.. py:class:: sponge.function.VelocityGenerator(temperature: float = 300, remove_translation: bool = True, seed: int = 0, seed2: int = 0, length_unit: str = None, energy_unit: str = None)

    根据温度产生系统中原子的速度。

    参数：
        - **temperature** (float) - 温度。默认值： ``300.0`` 。
        - **remove_translation** (bool) - 是否在基于周期性边界条件的情况下计算距离。默认值： ``True`` 。
        - **seed** (int) - 标准常态下的随机种子。默认值： ``0`` 。
        - **seed2** (int) - 标准常态下的随机种子2。默认值： ``0`` 。
        - **length_unit** (str) - 长度单位。默认值： ``None`` 。
        - **energy_unit** (str) - 能量单位。默认值： ``None`` 。

    .. py:method:: set_temperature(temperature: float)

        设定温度。

        参数：
            - **temperature** (float) - 温度值。

    .. py:method:: construct(shape: tuple, atom_mass: Tensor, mask: Tensor = None)

        随机生成系统中原子的速度。

        参数：
            - **shape** (tuple) - 速度的shape
            - **atom_mass** (Tensor) - 系统的原子质量。张量的shape为 :math:`(B, A)` ，数据类型为float。
                                       其中，B表示batchsize，例如，模拟中的步行者数量。A表示原子的数量。
            - **mask** (Tensor) - 原子的掩码。张量的shape为 :math:`(B, A)` ，数据类型为bool。默认值： ``None``。

        返回：
            Tensor。速度。张量的shape为 :math:`(B, A, D)` ，数据类型为float。其中D表示仿真系统的空间维度。通常为3。
