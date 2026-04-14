mindsponge.metrics.compute_renamed_ground_truth
===============================================

.. py:function:: mindsponge.metrics.compute_renamed_ground_truth(atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists, atom14_pred_positions, atom14_alt_gt_exists)

    由于蛋白质中部分氨基酸侧链存在对称性，因此可能存在等价的构象。本函数根据预测的蛋白质三维坐标与多种等价的构象作比较，并从中选出和预测坐标最接近的构象坐标作为结构标签。

    参数：
        - **atom14_gt_positions** (Tensor) - 按照稠密编码方式编码，蛋白质全原子坐标。shape :math:`(N_{res}, 14, 3)` 。
        - **atom14_alt_gt_positions** (Tensor) - 按稠密编码方式编码，根据对称性变换的等价全原子坐标。shape :math:`(N_{res}, 14, 3)` 。
        - **atom14_atom_is_ambiguous** (Tensor) - 由于部分氨基酸结构具有局部对称性，其对称原子编码可调换，具体原子参考 `common.residue_atom_renaming_swaps` 该特征记录了原子不确定的编码位置。shape :math:`(N_{res}, 14)` 。
        - **atom14_gt_exists** (Tensor) - 按照稠密编码方式编码，表示蛋白的相应原子在标签中是否存在的掩码。shape :math:`(N_{res}, 14)` 。
        - **atom14_pred_positions** (Tensor) - 按照稠密编码方式编码，预测的蛋白质全原子坐标。shape :math:`(N_{res}, 14, 3)` 。
        - **atom14_alt_gt_exists** (Tensor) - 按照稠密编码方式编码，表示蛋白的等价构象中的相应原子在标签中是否存在的掩码。shape :math:`(N_{res}, 14)` 。

    返回：
        - **alt_naming_is_better** (Tensor) - 记录了所有氨基酸是否在对称变换后更贴近预测坐标。shape :math:`(N_{res}, )` 。
        - **renamed_atom14_gt_positions** (Tensor) - 对称变换后重命名的，稠密编码的原子所对应的真实三维坐标。shape :math:`(N_{res}, 14, 3)` 。
        - **renamed_atom14_gt_exists** (Tensor) - 对称变换后重命名的、稠密编码的原子在标签中是否存在的掩码。shape :math:`(N_{res}, 14)` 。

    符号：
        :math:`N_{res}` - 蛋白质中氨基酸个数，按蛋白质一级序列排列。
