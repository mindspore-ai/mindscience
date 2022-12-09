mindsponge.potential.PotentialCell
==================================

.. py:class:: mindsponge.potential.PotentialCell(cutoff=None, exclude_index=None, length_unit=None, energy_unit=None, units=None, use_pbc=None)

    势能的基础单元。

    参数：
        - **cutoff** (float) - 中断距离。默认值："None"。
        - **exclude_index** (Tensor) - 应从无键相互作用中被排除的原子索引，shape为(B, A, Ex)，数据类型为int。默认值："None"。
        - **length_unit** (str) - 位置坐标的长度单位。默认值："None"。
        - **energy_unit** (str) - 能量单位。默认值："None"。
        - **units** (Units) - 长度和能量单位。默认值："None"。
        - **use_pbc** (bool, 可选) - 是否使用周期性边界条件。如果为None，则不使用周期性边界条件。默认值："None"。

    输出：
        Tensor。势，shape为(B, 1)。数据类型为float。

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