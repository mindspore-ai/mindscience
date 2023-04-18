mindflow.loss.RelativeRMSELoss
==============================

.. py:class:: mindflow.loss.RelativeRMSELoss(reduction="sum")

    相对均方根误差（RRMSE）是由均方根值归一化的均方根误差，其中每个残差都是根据实际值缩放的。
    Relative RMSELoss用来测量 :math:`x` 和 :math:`y` 之间的相对均方根误差，其中 :math:`x` 是预测值， :math:`y` 是目标值。

    假设 :math:`x` 和 :math:`y` 为一维Tensor，长度为 :math:`N` ，则 :math:`x` 和 :math:`y` 的损失为:

    .. math::
        loss = \sqrt{\frac{\sum_{i=1}^{N}{(x_i-y_i)^2}}{\sum_{i=1}^{N}{(y_i)^2}}}

    参数：
        - **reduction** (str) - `reduction` 决定了计算模式。有三种模式可选： ``"mean"``、 ``"sum"`` 和 ``"none"``。默认值： ``"sum"``。

    输入：
        - **prediction** (Tensor) - 网络模型预测值。Tensor的形状 :math:`(N, *)` 其中 :math:`*` 表示任意维度。
        - **labels** (Tensor) - 样本的真实值。Tensor的shape :math:`(N, *)` 其中 :math:`*` 表示任意维度，通常情况下和 `prediction` 的shape一致。但是，也支持labels的shape和prediction的shape不一致，两者应该可以相互广播。

    输出：
        Tensor。加权计算出的损失。