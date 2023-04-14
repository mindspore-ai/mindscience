mindsponge.metrics.local_distance_difference_test
===================================================

.. py:function:: mindsponge.metrics.local_distance_difference_test(predicted_points, true_points, true_points_mask, cutoff=15, per_residue=False)

    计算真实与预测的 :math:`C\alpha` 坐标的局部距离误差。
    首先分别计算真实和预测 :math:`C\alpha` 原子坐标的距离矩阵，
    :math:`D = (((x[None,:] - x[:,None])^2).sum(-1))^{0.5}`。
    然后计算两者差值小于固定数值的比例：
    :math:`lddt = (rate(abs(D_{true} - D_{pred}) < 0.5) + rate(abs(D_{true} - D_{pred}) < 1.0) + rate(abs(D_{true} - D_{pred}) < 2.0) + rate(abs(D_{true} - D_{pred}) < 4.0))/4`。

    `Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca" <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_.

    参数：
        - **predicted_points** (Tensor) - 预测的 :math:`C\alpha` 原子的坐标，shape为 :math:`(1, N_{res}, 3)` ，其中 :math:`N_{res}` 是蛋白质中的残基数目。
        - **true_points** (Tensor) - 真实的 :math:`C\alpha` 原子的坐标，shape为 :math:`(1, N_{res}, 3)` 。
        - **true_points_mask** (Tensor) - true_points的mask，shape为 :math:`(1, N_{res}, 1)` 。
        - **cutoff** (float) - 距离误差的截断点，超过该距离时梯度不再考虑，常量。
        - **per_residue** (bool) - 指示是否按残基为单位计算局部距离差，如果设为True则按残基为单位返回局部距离差值，默认值： ``False``。

    返回：
        - **score** (list) - Tensor。局部距离误差，如果 `per_residue` 为 ``False`` 则shape为 :math:`(1,)` ，否则为 :math:`(1, N_{res})` 。