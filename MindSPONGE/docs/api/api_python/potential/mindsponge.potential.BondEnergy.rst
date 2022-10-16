mindsponge.potential.BondEnergy
===============================

.. py:class:: mindsponge.potential.BondEnergy(index, force_constant, bond_length, parameters, use_pbc, length_unit="nm", energy_unit="kj/mol", units)

    键长的能量。

    参数：
        - **index** (Tensor) - 键的原子索引。
        - **force_constant** (Tensor) - 键长的谐和力常数。
        - **bond_length** (Tensor) - 键长的平衡值。
        - **parameters** (dict) - 力场参数。
        - **use_pbc** (bool) - 是否使用PBC。
        - **length_unit** (str) - 长度单位。默认值："nm"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。

    输出：
        Tensor。能量，shape为(B, 1)。

    符号：
        - **B** - Batch size。
        - **b** - 键的数量。
        - **D** - 模拟系统的维度。