mindsponge.common.get_pdb_info
==============================

.. py:function:: mindsponge.common.get_pdb_info(pdb_path)

    从pdb文件中获取原子坐标，残基序号等信息。针对蛋白质全原子坐标编码分为两种形式：分别为稀疏编码和稠密编码，详见：
    `common.make_atom14_positions`。本函数提供使用以上两种编码方式获取的蛋白质特征信息。

    参数：
        - **pdb_path** (str) - 输入pdb文件的路径。
  
    返回：
        dict，包含以下key值

        - **aatype** (numpy.array) 蛋白质一级序列编码，编码方式参考 `common.residue_constants.restype_order`， 取值范围 :math:`[0,20]` ，若为20表示该氨基酸为unkown（`UNK`）。 shape :math:`(N_{res}, )` 。
        - **all_atom_positions** (numpy.array) pdb文件对应蛋白质序列所有原子坐标。 shape :math:`(N_{res}, 37)` 。
        - **all_atom_mask** (numpy.array) 蛋白质所有原子坐标掩码。shape :math:`(N_{res}, 37)` ，若对应位置为0则表示该氨基酸不含该原子坐标。
        - **atom14_atom_exists** (numpy.array) 按照稠密编码方式编码，蛋白质全原子掩码，有原子位置为1，无原子位置为0。shape :math:`(N_{res}, 14)` 。
        - **atom14_gt_exists** (numpy.array) 按照稠密编码方式编码，蛋白质全原子掩码，默认与atom14_atom_exists一致。shape :math:`(N_{res}, 14)` 。
        - **atom14_gt_positions** (numpy.array) 按照稠密编码方式编码，蛋白质全原子坐标。shape :math:`(N_{res}, 14, 3)` 。
        - **residx_atom14_to_atom37** (numpy.array) 稀疏编码方式原子在稠密编码方式中的索引映射。shape :math:`(N_{res}, 14)` 。
        - **residx_atom37_to_atom14** (numpy.array) 稠密编码方式原子在稀疏编码方式中的索引映射。shape :math:`(N_{res}, 37)` 。
        - **atom37_atom_exists** (numpy.array) 按稀疏编码方式编码，蛋白质全原子掩码信息，有原子位置为1，无原子位置为0。shape :math:`(N_{res}, 37)` 。
        - **atom14_alt_gt_positions** (numpy.array) 按稠密编码方式编码，因手性蛋白对应全原子坐标。shape :math:`(N_{res}, 14, 3)` 。
        - **atom14_alt_gt_exists** (numpy.array) 按照稠密编码方式编码，对应手性蛋白全原子掩码。shape :math:`(N_{res}, 14)` 。
        - **atom14_atom_is_ambiguous** (numpy.array) 由于部分氨基酸结构具有局部对称性，其对称原子编码可调换，具体原子参考 `common.residue_atom_renaming_swaps` 该特征记录了原子不确定的编码位置。shape :math:`(N_{res}, 14)` 。
        - **residue_index** (numpy.array) 蛋白质序列编码index信息，大小从1到 :math:`N_{res}` 。shape :math:`(N_{res}, )` 。

    符号:
        - :math:`N_{res}` - 蛋白质中氨基酸个数，按蛋白质一级序列排列。
