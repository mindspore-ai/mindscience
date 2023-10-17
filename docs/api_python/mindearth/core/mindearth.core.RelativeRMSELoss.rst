mindearth.core.RelativeRMSELoss
================================

.. py:class:: mindearth.core.RelativeRMSELoss(reduction='mean')

    相对均方根误差（RRMSE）是由均方根值归一化的均方根误差，其中每个残差都是根据实际值缩放的。
    Relative RMSELoss用来测量 :math:`x` 和 :math:`y` 之间的相对均方根误差，其中 :math:`x` 是预测值， :math:`y` 是目标值。

    为简单起见，令 :math:`x` 和 :math:`y` 为长度为 :math:`N` 的一维Tensor， :math:`x` 和 :math:`y` 的损失如下：

    .. math::
        loss = \sqrt{\frac{\frac{1}{N}\sum_{i=1}^{N}{(x_i-y_i)^2}}{sum_{i=1}^{N}{(y_i)^2}}}

    参数：
        - **reduction** (str) - `reduction` 决定了计算模式。有三种模式可选： ``"mean"``、 ``"sum"`` 和 ``"none"``。默认值： ``"mean"``。

    输入：
        - **prediction** (Tensor) - 预测值，公式中的 :math:`x` ，shape为 :math:`(N, *)` 的Tensor， :math:`*` 代表任意数量的其他维度。
        - **labels** (Tensor) - 样本的真实值，公式中的 :math:`y` 。Tensor的shape :math:`(N, *)` 其中 :math:`*` 表示任意维度，通常情况下和 `prediction` 的shape一致。但是，也支持 `labels` 的shape和 `prediction` 的shape不一致，两者需满足可相互广播。

    输出：
        Tensor，加权损失浮点数。

        - **output** (Tensor) - shape为 :math:`()` 的Tensor。