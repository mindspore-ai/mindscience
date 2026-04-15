mindsponge.common.make_atom14_positions
=======================================

.. py:function:: mindsponge.common.make_atom14_positions(aatype, all_atom_mask, all_atom_positions)

    本函数提供将稀疏编码方式转为稠密编码方式的功能。

    针对蛋白质全原子坐标编码分为两种形式

    - 稀疏编码：20种氨基酸包含原子种类共计37种，详见 `common.residue_constants.atom_types` ，故可将蛋白质全原子坐标编码为 :math:`(N_{res}, 37, 3)` 的张量。

    - 稠密编码：单氨基酸最多包含14种不同的原子类型，详见 `common.residue_constants.restype_name_to_atom14_names` ，故可将蛋白质全原子坐标编码为 :math:`(N_{res}, 14, 3)` 的张量。

    参数：
        - **aatype** (numpy.ndarray) - 蛋白质一级序列编码，编码方式参考 `common.residue_constants.restype_order`, 取值范围 :math:`[0,20]` ，若为20表示该氨基酸为unkown（`UNK`）。
        - **all_atom_mask** (numpy.ndarray) - 蛋白质所有原子坐标掩码，维度为 :math:`(N_{res}, 37)` ，若对应位置为0则表示该氨基酸不含该原子坐标。
        - **all_atom_positions** (numpy.ndarray) - 蛋白质所有原子坐标，维度为 :math:`(N_{res}, 37, 3)` 。

    返回：
        - numpy.array。按照稠密编码方式编码，蛋白质全原子掩码，包含unkown氨基酸原子， :math:`(N_{res}, 14)` 。
        - numpy.array。按照稠密编码方式编码，蛋白质全原子掩码，不包含unkown氨基酸原子， :math:`(N_{res}, 14)` 。
        - numpy.array。按照稠密编码方式编码，蛋白质全原子坐标,  :math:`(N_{res}, 14, 3)` 。
        - numpy.array。稀疏编码方式原子在稠密编码方式中的索引映射， :math:`(N_{res}, 14)` 。
        - numpy.array。稠密编码方式原子在稀疏编码方式中的索引映射， :math:`(N_{res}, 37)` 。
        - numpy.array。按照稀疏编码方式编码，蛋白质全原子掩码，包含unkown氨基酸原子， :math:`(N_{res}, 37)` 。
        - numpy.array。针对稠密编码方式全原子坐标进行手性变换后的全原子坐标。 :math:`(N_{res}, 14, 3)` 。
        - numpy.array。手性变换后原子掩码， :math:`(N_{res}, 14)` 。
        - numpy.array。进行手性变换的原子标识，1为进行变换，0为未进行变换， :math:`(N_{res}, 14)` 。

    符号：
        - :math:`N_{res}` - 蛋白质中氨基酸个数，按蛋白质一级序列排列。