mindsponge.potential.PotentialCell
==================================

.. py:class:: mindsponge.potential.PotentialCell(cutoff, exclude_index, length_unit, energy_unit, units, use_pbc)

    势能的基础单元。

    参数：
        - **cutoff** (float) - 中断距离。
        - **exclude_index** (Tensor) - 应从无键相互作用中被排除的原子索引。
        - **length_unit** (str) - 位置坐标的长度单位。
        - **energy_unit** (str) - 能量单位。
        - **units** (Units) - 长度和能量单位。
        - **use_pbc** (bool, 可选) - 是否使用周期性边界条件。

    输出：
        Tensor。势。

    .. py:method:: exclude_index()

        排除索引。

        返回：
            Tensor。排除索引。

    .. py:method:: set_cutoff(cutoff)

        设置中断距离。

        参数：
            - **cutoff** (Tensor) - 中断距离。

    .. py:method:: set_exclude_index(exclude_index)

        设置排除索引。

        参数：
            - **exclude_index** (Tensor) - 应该从非键相互作用中被排除的原子的索引。

    .. py:method:: set_pbc(use_pbc)

        设置是否使用周期性边界条件PBC。

        参数：
            - **use_pbc** (bool, 可选) - 是否使用PBC。