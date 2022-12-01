mindsponge.common.find_optimal_renaming
=======================================

.. py:function:: mindsponge.common.find_optimal_renaming(atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists, atom14_pred_positions)

    通过最大化LDDT值，确定具手性的氨基酸残基的最佳原子坐标。具体计算过程参考： `Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms" <https://www.nature.com/articles/s41586-021-03819-2>`_ 。
    

    参数：
        - **atom14_gt_positions** (Tensor) - 全局坐标系中的真实原子坐标值，采用atom14表示，shape为 :math:`(N_{res}, 14, 3)`。
        - **atom14_alt_gt_positions** (Tensor) - 备选的真实原子坐标，部分氨基酸存在手性，因此使用手性对称的位置作为备选真实原子坐标值，shape为 :math:`(N_{res}, 14, 3`)。
        - **atom14_atom_is_ambiguous** (Tensor) - 掩码表示是否是手性原子中，shape为 :math:`(N_{res}, 14)` 。
        - **atom14_gt_exists** (Tensor) - 掩码表明真实结构中原子是否存在，shape为 :math:`(N_{res}, 14)` 。
        - **atom14_pred_positions** (Tensor) - 全局坐标系中预测得到的原子坐标值，shape为 :math:`(N_{res}, 14, 3)` 。

    返回：
        Tensor。 `atom14_alt_gt_positions` 与  `atom14_pred_positions` 更接近的位置为1，否则是0，shape为 :math:`(N_{res},)` 。