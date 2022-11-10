mindsponge.potential.BondEnergy
===============================

.. py:class:: mindsponge.potential.BondEnergy(index=None, force_constant=None, bond_length=None, parameters=None, use_pbc=None, length_unit="nm", energy_unit="kj/mol", units=None)

    键长的能量。

    .. Math::

        E_{bond}(b_{ij}) = 1 / 2 * k_{ij}^b * (b_{ij} - b_{ij}^0) ^ 2

    参数：
        - **index** (Tensor) - 键的原子索引，数据类型为int。默认值："None"。
        - **force_constant** (Tensor) - 键长的谐和力常数，数据类型为float。默认值："None"。
        - **bond_length** (Tensor) - 键长的平衡值，数据类型为float。默认值："None"。
        - **parameters** (dict) - 力场参数。默认值："None"。
        - **use_pbc** (bool) - 是否使用PBC。默认值："None"。
        - **length_unit** (str) - 长度单位。默认值："nm"。
        - **energy_unit** (str) - 能量单位。默认值："kj/mol"。
        - **units** (Units) - 长度和能量单位。默认值："None"。

    输出：
        Tensor。能量，shape为(B, 1)，数据类型为float。

    符号：
        - **B** - Batch size。
        - **b** - 键的数量。
        - **D** - 模拟系统的维度，一般为3。