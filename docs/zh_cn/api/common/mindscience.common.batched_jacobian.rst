mindscience.common.batched_jacobian
====================================

.. py:function:: mindscience.common.batched_jacobian(model)

    计算网络模型的雅可比矩阵。

    .. note::
        本函数在实现中使用了 ``mindspore.jacrev`` 接口来计算 Jacobian 矩阵，因此要求 **MindSpore 版本 >= 2.0.0**。

    参数：
        - **model** (mindspore.nn.Cell) - 输入维度为 in_channels 输出维度为 out_channels 的网络模型。

    返回：
        Tensor，用于计算雅可比矩阵的Jacobian实例。输入维度为 :math:`[batch_size，in_channels]`，输出维度为 :math:`[out_channels，batch_size，in_channels]`。
