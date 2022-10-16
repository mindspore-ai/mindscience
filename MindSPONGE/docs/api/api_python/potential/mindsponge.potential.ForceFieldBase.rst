mindsponge.potential.ForceFieldBase
===================================

.. py:class:: mindsponge.potential.ForceFieldBase(energy, cutoff, exclude_index, length_unit, energy_unit, units, use_pbc)

    力场的基础层。

    参数：
        - **energy** (Union[EnergyCell, list]) - 能量项。
        - **cutoff** (float) - 中断距离。
        - **exclude_index** (Tensor) - 需要从邻居列表中排除的原子的索引。
        - **length_unit** (str) - 位置坐标的长度单位。
        - **energy_unit** (str) - 能量单位。
        - **units** (Units) - 长度和能量单位。
        - **use_pbc** (bool, 可选) - 是否使用PBC。

    输出：
        Tensor。势，shape为(B, 1)。

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

    .. py:method:: set_pbc(use_pbc)

        设置是否使用周期性边界条件PBC。

        参数：
            - **use_pbc** (bool, 可选) - 是否使用PBC。

    .. py:method:: set_unit_scale()

        设置单位范围。

        返回：
            Tensor。输出单位范围。

    .. py:method:: set_units(length_unit, energy_unit, units)

        设置单位。

        参数：
            - **length_unit** (str) - 位置坐标的长度单位。
            - **energy_unit** (str) - 能量单位。
            - **units** (Units) - 长度和能量单位。