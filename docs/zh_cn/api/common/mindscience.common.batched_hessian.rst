mindscience.common.batched_hessian
===================================

.. py:function:: mindscience.common.batched_hessian(model)

    计算网络模型的海森矩阵。

    参数：
        - **model** (mindspore.nn.Cell) - 输入维度为 in_channels 输出维度为 out_channels 的网络模型。

    返回：
        Tensor，用于计算海森矩阵的 Hessian 实例。输入维度为 :math:`[batch_size，in_channels]` ，输出维度为 :math:`[out_channels，in_channels，batch_size，in_channels]`。
    
    .. note::
        使用 `mindspore.jacrev` 所在的 MindSpore 版本必须 >= 2.0.0。
