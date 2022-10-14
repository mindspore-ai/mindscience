mindsponge.common.make_atom14_positions
=======================================

.. py:class:: mindsponge.common.make_atom14_positions(aatype, all_atom_mask, all_atom_positions)

    构建更稠密的原子位置(14维度而不是37维)。

    参数：
        - **aatype** (numpy.array) - 氨基酸的类型。
        - **all_atom_mask** (numpy.array) - 输入对应的mask。
        - **all_atom_positions** (numpy.array) - 所有原子的位置。

    输出：
        - numpy.array。atom14位置对应的mask。
        - numpy.array。真实的atom14对应位置的mask。
        - numpy.array。真实的atom14对应位置。
        - numpy.array。把atom14的残基序号map成atom37的残基。
        - numpy.array。反向mapping时获取切片。
        - numpy.array。atom37位置对应的mask。
        - numpy.array。提供把给予的残基序列转换为真实位置的转换矩阵。
        - numpy.array。可替换真实值的mask。
        - numpy.array。针对序列生成一个不确定的mask。