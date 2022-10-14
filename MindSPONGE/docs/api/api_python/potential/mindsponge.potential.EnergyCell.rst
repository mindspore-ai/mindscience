mindsponge.potential.EnergyCell
===============================

.. py:class:: mindsponge.potential.EnergyCell(label, output_dim=1, length_unit="nm", energy_unit="kj/mol", units, use_pbc)

    能量项的基础层。

    参数：
        - **label** (str) - 能量的标签名称。
        - **output_dim** (int) - 输出维度。默认值：1。
        - **length_unit** (str) - 位置坐标的长度单位。默认值："nm"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。
        - **use_pbc** (bool) - 是否使用PBC。

    输出：
        Tensor。能量，shape为(B, 1)。

    .. py:method:: convert_energy_from(unit)

        将能量从外部单元转换到内部单元。

        参数：
            - **unit** (str) - 长度和能量的单位。

        返回：
            float。从外部单元转换到内部单元的能量。

    .. py:method:: convert_energy_to(unit)

        将能量从内部单元转换到外部单元。

        参数：
            - **unit** (str) - 长度和能量的单位。

        返回：
            float。从内部单元转换到外部单元的能量。

    .. py:method:: set_cutoff(cutoff)

        设置中断距离。

        参数：
            - **cutoff** (float) - 中断距离。

    .. py:method:: set_input_unit(units)

        设置输入坐标的长度单位。

        参数：
            - **units** (Units) - 长度和能量的单位。

    .. py:method:: set_pbc(use_pbc)

        设置是否使用PBC。

        参数：
            - **use_pbc** (bool, 可选) - 是否使用PBC。