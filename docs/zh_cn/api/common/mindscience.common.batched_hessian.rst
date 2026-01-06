mindscience.common.batched_hessian
===================================

.. py:function:: mindscience.common.batched_hessian(model)

    计算网络模型的海森矩阵。

    .. note::
        本函数在实现中使用了 ``mindspore.jacrev`` 接口来计算 Hessian 矩阵，因此要求 **MindSpore 版本 >= 2.0.0**。

    参数：
        - **model** (mindspore.nn.Cell) - 输入维度为 in_channels 输出维度为 out_channels 的网络模型。

    返回：
        Tensor，用于计算海森矩阵的 Hessian 实例。输入维度为 :math:`[batch_size, in_channels]` ，输出维度为 :math:`[out_channels, in_channels, batch_size, in_channels]`。
    