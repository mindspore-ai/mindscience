mindsponge.metrics.between_residue_clash
========================================

.. py:function:: mindsponge.metrics.between_residue_clash(atom14_pred_positions, atom14_atom_exists, atom14_atom_radius, residue_index, c_one_hot, n_one_hot, overlap_tolerance_soft, overlap_tolerance_hard, cys_sg_idx)

    该函数用于计算属于蛋白质序列不同残基的原子对在空间中由于位置过于接近而存在的冲突损失。（针对蛋白质全原子坐标编码分为两种形式：分别为稀疏编码和稠密编码，详见： `common.make_atom14_positions` ）

    参数：
        - **atom14_pred_positions** (Tensor) - 以稠密编码方式编码的蛋白质全所有原子坐标。shape :math:`(N_{res}, 14, 3)` 。
        - **atom14_atom_exists** (Tensor) - 按照稠密编码方式编码，蛋白质全原子mask，有原子位置为1，无原子位置为0。shape :math:`(N_{res}, 14)` 。
        - **atom14_atom_radius** (Tensor) - 按照稠密编码方式编码，每个原子的范德华半径。shape :math:`(N_{res}, 14)` 。
        - **residue_index** (Tensor) - 蛋白质序列编码index信息，大小从1到 :math:`N_{res}` 。shape :math:`(N_{res}, )` 。
        - **c_one_hot** (Tensor) - 按照稠密编码方式对C原子的独热编码。shape: `(14, )` 。
        - **n_one_hot** (Tensor) - 按照稠密编码方式对N原子的独热编码。shape: `(14, )` 。
        - **overlap_tolerance_soft** (float) - 空间中原子由于过近会导致重叠，该系数表示原子间空间距离的软容忍数。
        - **overlap_tolerance_hard** (float) - 空间中原子由于过近会导致重叠，该系数表示原子间空间距离的硬容忍数。
        - **cys_sg_idx** (Tensor) - 半胱氨酸在蛋白质编码中的index，详见： `mindsponge.common.residue_constants` 。 shape: `( )` 。

    返回：
        - **mean_loss** (Tensor) - 在空间中，所有原子间距离中超出原子范德化半径的总平均距离损失。即含有距离冲突的原子平均损失。shape: `( )`
        - **per_atom_loss_sum** (Tensor) - 在空间中，所有原子间距离中超出原子范德化半径的总和除以总原子个数。即平均单个原子距离损失（包括没有距离冲突的原子）。shape :math:`(N_{res}, 14)` 。
        - **per_atom_clash_mask** (Tensor) - 在空间中，所有原子间距离中超出原子范德化半径的原子mask。 ``1`` 表示有冲突， ``0`` 表示没有冲突。shape :math:`(N_{res}, 14)` 。

    符号:
        - :math:`N_{res}` - 蛋白质中氨基酸个数，按蛋白质一级序列排列。
