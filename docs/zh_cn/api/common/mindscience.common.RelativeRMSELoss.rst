mindscience.common.RelativeRMSELoss
====================================

.. py:class:: mindscience.common.RelativeRMSELoss(reduction="sum")

    相对均方根误差（RRMSE）是对均方根误差进行归一化后的结果，其归一化因子为真实值的均方根，其中每一个残差项都相对于真实值进行缩放。
    相对均方根误差用于逐元素地度量预测值 :math:`x` 与标签 :math:`y` 之间的均方根误差，
    其中 :math:`x` 表示预测值，:math:`y` 表示标签。

    为简化描述，设 :math:`x` 和 :math:`y` 为长度为 :math:`N` 的一维 Tensor，则 :math:`x` 与 :math:`y` 的损失定义如下：

    .. math::
        loss = \sqrt{\frac{\sum_{i=1}^{N}{(x_i-y_i)^2}}{\sum_{i=1}^{N}{(y_i)^2}}}

    参数：
        - **reduction** (str, 可选) - 应用于损失的归约方式。支持 ``"mean"``、 ``"sum"`` 或 ``"none"``。默认 ``"sum"``。

    输入：
        - **prediction** (Tensor) - 网络的预测值。形状为 :math:`(N, *)` 的 Tensor，其中 :math:`*` 表示任意数量的附加维度。
        - **labels** (Tensor) - 样本的真实值。形状为 :math:`(N, *)` 的张量，其中 :math:`*` 表示任意数量的附加维度，在常见情况下与 `prediction` 具有相同的形状。
          该接口支持 `labels` 的形状与 `prediction` 不同，但二者需要能够相互广播。

    输出：
        - **output** (Tensor) - 加权后的损失值。
