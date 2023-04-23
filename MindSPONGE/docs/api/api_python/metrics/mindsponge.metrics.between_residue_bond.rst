mindsponge.metrics.between_residue_bond
=======================================

.. py:function:: mindsponge.metrics.between_residue_bond(pred_atom_positions, pred_atom_mask, residue_index, aatype, tolerance_factor_soft=12.0, tolerance_factor_hard=12.0)

    该函数主要用于计算序列上连续氨基酸残基之间是否存在肽键长度及角度冲突，可以对蛋白质结构冲突进行惩罚。（针对蛋白质全原子坐标编码分为两种形式：分别为稀疏编码和稠密编码，详见：`common.make_atom14_positions` ）

    参数：
        - **pred_atom_positions** (Tensor) - 以稠密或稀疏编码方式编码的蛋白质所有原子三维坐标，shape :math:`(N_{res}, 37, 3)`（稀疏编码）或 :math:`(N_{res}, 14, 3)` （稠密编码）。
        - **pred_atom_mask** (Tensor) - 以稠密或稀疏编码方式编码的蛋白质全原子mask， shape :math:`(N_{res}, 37)`（稀疏编码）或 :math:`(N_{res}, 14)` （稠密编码）。
        - **residue_index** (Tensor) - 蛋白质序列编码index信息，大小从1到 :math:`N_{res}` 。shape :math:`(N_{res}, )` 。
        - **aatype** (Tensor) - 蛋白质一级序列编码，编码方式参考 `common.residue_constants.restype_order`，取值范围 :math:`[0,20]` ，若为20表示该氨基酸为unkown（`UNK`）。 shape :math:`(N_{res}, )` 。
        - **tolerance_factor_soft** (float) - 根据蛋白质结构分布的标准偏差测量的软公差因子，默认为 ``12.0`` 。
        - **tolerance_factor_hard** (float) - 根据蛋白质结构分布的标准偏差测量的硬公差因子，默认为 ``12.0`` 。

    返回：
        - **c_n_loss_mean** (Tensor) C-N 肽键长度冲突损失。shape: :math:`( )` 。
        - **ca_c_n_loss_mean** (Tensor) CA-C-N键角的冲突损失。shape: :math:`( )` 。
        - **c_n_ca_loss_mean** (Tensor) C-N-CA键角的冲突损失。shape: :math:`( )` 。
        - **per_residue_loss_sum** (Tensor) 所有氨基酸残基的冲突损失总和，包括键角键长损失。shape :math:`(N_{res}, )` 。
        - **per_residue_violation_mask** (Tensor) 指示每个氨基酸残基是否存在键长或键角损失；1代表存在，0代表不存在。shape :math:`(N_{res}, )` 。

    符号：
        - :math:`N_{res}` - 蛋白质中氨基酸个数，按蛋白质一级序列排列。
