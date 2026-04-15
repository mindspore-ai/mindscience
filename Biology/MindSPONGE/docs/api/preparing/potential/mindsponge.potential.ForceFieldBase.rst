mindsponge.potential.ForceFieldBase
===================================

.. py:class:: mindsponge.potential.ForceFieldBase(energy=None, cutoff=None, exclude_index=None, length_unit=None, energy_unit=None, units=None, use_pbc=None)

    力场的基础层。

    参数：
        - **energy** (Union[EnergyCell, list]) - 能量项。默认值："None"。
        - **cutoff** (float) - 中断距离。默认值："None"。
        - **exclude_index** (Tensor) - 需要从邻居列表中排除的原子的索引。默认值："None"。
        - **length_unit** (str) - 位置坐标的长度单位。默认值："None"。
        - **energy_unit** (str) - 能量单位。默认值："None"。
        - **units** (Units) - 长度和能量单位。默认值："None"。
        - **use_pbc** (bool, 可选) - 是否使用PBC。默认值："None"。

    输出：
        Tensor。势，shape为(B, 1)。数据类型为float。

    .. py:method:: set_cutoff(cutoff)

        设置中断距离。

        参数：
            - **cutoff** (Tensor) - 中断距离。

    .. py:method:: set_energy_cell(energy)

        设置能量。

        参数：
            - **energy** (Union[EnergyCell, list]) - 能量项。

        返回：
            层列表。

    .. py:method:: set_energy_scale(scale)

        设置能量范围。

        参数：
            - **scale** (Tensor) - 用于设置能量范围。

    .. py:method:: set_pbc(use_pbc=None)

        设置是否使用周期性边界条件PBC。

        参数：
            - **use_pbc** (bool, 可选) - 是否使用PBC。默认值："None"。

    .. py:method:: set_unit_scale()

        设置单位范围。

        返回：
            Tensor。输出单位范围。

    .. py:method:: set_units(length_unit=None, energy_unit=None, units=None)

        设置单位。

        参数：
            - **length_unit** (str) - 位置坐标的长度单位。默认值："None"。
            - **energy_unit** (str) - 能量单位。默认值："None"。
            - **units** (Units) - 长度和能量单位。默认值："None"。