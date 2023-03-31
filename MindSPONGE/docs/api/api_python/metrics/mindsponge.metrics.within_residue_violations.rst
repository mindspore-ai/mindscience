mindsponge.metrics.within_residue_violations
============================================

.. py:function:: mindsponge.metrics.within_residue_violations(atom14_pred_positions, atom14_atom_exists, atom14_dists_lower_bound, atom14_dists_upper_bound, tighten_bounds_for_loss, dists_mask_i)

    该函数用于计算蛋白质序列中同一个氨基酸下所有原子在空间中由于位置过于接近而存在的冲突损失。（针对蛋白质全原子坐标编码分为两种形式：分别为稀疏编码和稠密编码，详见：`common.make_atom14_positions` ）

    参数：
        - **atom14_pred_positions** (Tensor) - 以稠密编码方式编码的蛋白质所有原子三维坐标，shape :math:`(N_{res}, 14, 3)` 。
        - **atom14_atom_exists** (Tensor) - 按照稠密编码方式编码，蛋白质全原子掩码，有原子位置为1，无原子位置为0。shape :math:`(N_{res}, 14)` 。
        - **atom14_dists_lower_bound** (Tensor) - 按稠密编码方式编码的原子间允许的最小距离。shape :math:`(N_{res}, 14, 14)` 。
        - **atom14_dists_upper_bound** (Tensor) - 按稠密编码方式编码的原子间允许的最大距离。shape :math:`(N_{res}, 14, 14)` 。
        - **tighten_bounds_for_loss** (float) - 原子间距离冲突系数。
        - **dists_mask_i** (Tensor) - 以稠密编码方式编码的原子距离掩码矩阵。shape :math:`(14, 14)` 。

    返回：
        - **per_atom_loss_sum** (Tensor) 每个原子的总距离冲突误差。shape :math:`(N_{res}, 14)` 。
        - **per_atom_violations** (Tensor) 每个原子的冲突误差（键长和键角冲突最大值）。shape :math:`(N_{res}, 14)` 。

    符号：
        :math:`N_{res}` - 蛋白质中氨基酸个数，按蛋白质一级序列排列。
