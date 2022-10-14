mindsponge.potential.ForceField
===============================

.. py:class:: mindsponge.potential.ForceField(system, parameters, cutoff, length_unit, energy_unit, units)

    经典力场的势。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **parameters** (Union[dict, str]) - 力场参数。
        - **cutoff** (float) - 中断距离。
        - **length_unit** (str) - 位置坐标的长度单位。
        - **energy_unit** (str) - 能量单位。
        - **units** (Units) - 长度和能量单位。