mindsponge.function.VelocityGenerator
=====================================

.. py:class:: mindsponge.function.VelocityGenerator(temperature=300, remove_translation=True, seed=0, seed2=0, length_unit, energy_unit)

    根据温度产生系统中原子的速度。

    参数：
        - **temperature** (float) - 温度。默认值：300。
        - **remove_translation** (bool) - 是否在基于周期性边界条件的情况下计算距离。默认值：True。
        - **seed** (int) - 标准常态下的随机种子。默认值：0。
        - **seed2** (int) - 标准常态下的随机种子2。默认值：0。
        - **length_unit** (str) - 长度单位。
        - **energy_unit** (str) - 能量单位。

    .. py:method:: set_temperature(temperature)

        设定温度。

        参数：
            - **temperature** (float) -温度值。