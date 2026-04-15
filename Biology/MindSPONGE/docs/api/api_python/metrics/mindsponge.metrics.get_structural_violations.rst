mindsponge.metrics.get_structural_violations
============================================

.. py:function:: mindsponge.metrics.get_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37, atom14_pred_positions, violation_tolerance_factor=VIOLATION_TOLERANCE_ACTOR, clash_overlap_tolerance=CLASH_OVERLAP_TOLERANCE, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND, atomtype_radius=ATOMTYPE_RADIUS, c_one_hot=C_ONE_HOT, n_one_hot=N_ONE_HOT, dists_mask_i=DISTS_MASK_I, cys_sg_idx=CYS_SG_IDX)

    该函数主要用于计算蛋白序列中相邻的氨基酸是否存在不合理的肽键构型，以及不同原子是否在空间中距离过近，从而对蛋白质结构中不合理的空间冲突进行惩罚。（针对蛋白质全原子坐标编码分为两种形式：分别为稀疏编码和稠密编码，详见： `common.make_atom14_positions` ）

    参数：
        - **atom14_atom_exists** (Tensor) - 按照稠密编码方式编码，蛋白质全原子mask，有原子位置为1，无原子位置为0。shape :math:`(N_{res}, 14)` 。
        - **residue_index** (Tensor) - 蛋白质序列编码index信息，大小从0到 :math:`N_{res} - 1` 。shape :math:`(N_{res}, )` 。
        - **aatype** (Tensor) - 蛋白质一级序列编码，编码方式参考 `common.residue_constants.restype_order`，取值范围 :math:`[0,20]` ，若为20表示该氨基酸为unkown（`UNK`）。 shape :math:`(N_{res}, )` 。
        - **residx_atom14_to_atom37** (Tensor) - 稠密编码方式原子在稠密编码方式中的索引映射。shape :math:`(N_{res}, 14)` 。
        - **atom14_pred_positions** (Tensor) - 以稠密编码方式编码的蛋白质全所有原子坐标。shape :math:`(N_{res}, 14, 3)` 。
        - **violation_tolerance_factor** (float) - 氨基酸残基中原子间距离由于过近或过远会导致原子存在冲突，该系数为氨基酸残基内原子间距离容忍数。
        - **clash_overlap_tolerance** (float) - 空间中原子由于过近会导致重叠，该系数表示原子间空间距离的容忍数。
        - **lower_bound** (Tensor) - 按稠密编码方式编码的原子间允许的最小距离。shape :math:`(N_{res}, 14, 14)` 。
        - **upper_bound** (Tensor) - 按稠密编码方式编码的原子间允许的最大距离。shape :math:`(N_{res}, 14, 14)` 。
        - **atomtype_radius** (Tensor) - 按照稠密编码方式编码，每个原子的范德华半径。shape :math:`(37, )` 。
        - **c_one_hot** (Tensor) - 按照稠密编码方式对C原子的独热编码。shape: :math:`(14, )` 。
        - **n_one_hot** (Tensor) - 按照稠密编码方式对N原子的独热编码。shape: :math:`(14, )` 。
        - **dists_mask_i** (Tensor) - 按稠密编码方式编码的原子间距离矩阵。shape: :math:`(14, 14)` 。
        - **cys_sg_idx** (Tensor) - 半胱氨酸在蛋白质编码中的index，详见： `mindsponge.common.residue_constants` 。 shape: :math:`( )` 。

    返回：
        - **bonds_c_n_loss_mean** (Tensor) - C-N 肽键长度冲突的平均损失。shape: :math:`()` 。
        - **angles_ca_c_n_loss_mean** (Tensor) - CA-C-N键角冲突的平均损失。shape: :math:`()` 。
        - **angles_c_n_ca_loss_mean** (Tensor) - C-N-CA键角冲突的平均损失。shape: :math:`()` 。
        - **connections_per_residue_loss_sum** (Tensor) - 所有氨基酸残基关于肽键的冲突损失总和，包括键角键长损失。shape :math:`(N_{res}, )` 。
        - **connections_per_residue_violation_mask** (Tensor) - mask表示每个氨基酸残基是否存在键长或键角损失。shape :math:`(N_{res}, )` 。
        - **clashes_mean_loss** (Tensor) - 在空间中，所有原子间距离中超出原子范德化半径的总平均距离损失。即含有距离冲突的原子平均损失。shape :math:`()`
        - **clashes_per_atom_loss_sum** (Tensor) - 在空间中，所有原子间距离中超出原子范德化半径的总和除以总原子个数。即平均单个原子距离损失（包括没有距离冲突的原子）。shape :math:`(N_{res}, 14)` 。
        - **clashes_per_atom_clash_mask** (Tensor) - 在空间中，所有原子间距离中超出原子范德化半径的原子mask。1表示有冲突，0表示没有冲突。shape :math:`(N_{res}, 14)` 。
        - **per_atom_loss_sum** (Tensor) - 每个原子的总距离冲突误差。shape :math:`(N_{res}, 14)` 。
        - **per_atom_violations** (Tensor) - 每个原子的冲突误差（键长和键角冲突最大值）。shape :math:`(N_{res}, 14)` 。
        - **total_per_residue_violations_mask** (Tensor) - 氨基酸残基是否存在原子冲突掩码。 shape :math:`(N_{res}, )` 。
        - **structure_violation_loss** (Tensor) - 所有氨基酸残基原子冲突损失总和。 shape： :math:`()` 。

    符号:
        :math:`N_{res}` - 蛋白质中氨基酸个数，按蛋白质一级序列排列。