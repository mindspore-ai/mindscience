mindsponge.potential.ForceField
===============================

.. py:class:: mindsponge.potential.ForceField(system, parameters, cutoff=None, length_unit=None, energy_unit=None, units=None)

    经典力场的势。

    参数：
        - **system** (Molecule) - 模拟系统。
        - **parameters** (Union[dict, str]) - 力场参数。
        - **cutoff** (float) - 中断距离。默认值："None"。
        - **length_unit** (str) - 位置坐标的长度单位。默认值："None"。
        - **energy_unit** (str) - 能量单位。默认值："None"。
        - **units** (Units) - 长度和能量单位。默认值："None"。