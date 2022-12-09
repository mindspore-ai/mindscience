mindsponge.potential.NonbondEnergy
==================================

.. py:class:: mindsponge.potential.NonbondEnergy(label, output_dim=1, cutoff=None, length_unit="nm", energy_unit="kj/mol", use_pbc=None, units=None)

    非键能项的基本单元。

    参数：
        - **label** (str) - 能量的标签名称。
        - **output_dim** (int) - 输出维度。默认值：1。
        - **cutoff** (float) - 中断距离。默认值："None"。
        - **length_unit** (str) - 长度单位。默认值："nm"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **use_pbc** (bool) - 是否使用PBC。默认值："None"。
        - **units** (Units) - 长度和能量单位。默认值："None"。

    输出：
        Tensor。能量，shape为(B, 1),数据类型为float。

    .. py:method:: set_input_unit(units)

        对输入坐标设置长度单位。

        参数：
            - **units** (Units) - 长度和能量单位。