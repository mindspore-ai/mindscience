mindsponge.potential.NonbondEnergy
==================================

.. py:class:: mindsponge.potential.NonbondEnergy(label, output_dim=1, cutoff, length_unit="nm", energy_unit="kj/mol", use_pbc, units)

    非键能项的基本单元。

    参数：
        - **label** (str) - 能量的标签名称。
        - **output_dim** (float) - 输出维度。默认值：1。
        - **cutoff** (int) - 中断距离。
        - **length_unit** (bool) - 长度单位。默认值："nm"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **use_pbc** (str) - 是否使用PBC。
        - **units** (Units) - 长度和能量单位。

    输出：
        Tensor。能量，shape为(B, 1)。

    .. py:method:: set_input_unit(units)

        对输入坐标设置长度单位。

        参数：
            - **units** (Units) - 长度和能量单位。