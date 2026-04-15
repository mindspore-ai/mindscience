mindsponge.metrics.supervised_chi
=================================

.. py:function:: mindsponge.metrics.supervised_chi(sequence_mask, aatype, sin_cos_true_chi, torsion_angle_mask, sin_cos_pred_chi, sin_cos_unnormalized_pred, chi_weight, angle_norm_weight, chi_pi_periodic)

    计算主链与侧链扭转角的误差，扭转角用角度的正弦与余弦值表示，该误差由两项组成，第一项是正则化后的预测正弦余弦值与真实值的角度差，第二项是预测值的模量与1的差值，称为角度模量误差。
    `Jumper et al. (2021) Suppl. Alg. 27 "torsionAngleLoss"
    <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/
    MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_。

    参数：
        - **sequence_mask** (Tensor) - 序列残基的mask，shape为 :math:`(N_{res},)` ，其中 :math:`N_{res}` 是蛋白质中的残基数目。
        - **aatype** (Tensor) - 序列中的氨基酸残基类型，shape为 :math:`(N_{res},)` 。
        - **sin_cos_true_chi** (Tensor) - shape为 :math:`(N_{res}, 14)` ，扭转角的正弦和余弦值，每个氨基酸残基有七个扭转角，其中主链三个，侧链四个。
        - **torsion_angle_mask** (Tensor) - 侧链扭转角的mask，shape为 :math:`(N_{res}, 4)` 。
        - **sin_cos_pred_chi** (Tensor) - shape为 :math:`(N_{res}, 4, 2)` ，预测的侧链扭转角的正弦和余弦值。
        - **sin_cos_unnormalized_pred** (Tensor) - 预测的扭转角的正弦和余弦值，没有做过正则化，shape为 :math:`(N_{recycle}, N_{res}, 7, 2)` ，其中 :math:`N_{recycle}` 是Structure模块中的循环次数。
        - **chi_weight** (float) - 角度差损失函数项的权重。
        - **angle_norm_weight** (float) - 角度模量损失函数项的权重。
        - **chi_pi_periodic** (Tensor) - 扭转角的周期性信息，某些氨基酸的某些扭转角具有周期性。氨基酸性质的常量，shape是 :math:`(21, 4)` ，21代表二十种氨基酸加未知氨基酸。

    返回：
        Tensor，主链与侧链扭转角的误差，shape为 :math:`()` 。
