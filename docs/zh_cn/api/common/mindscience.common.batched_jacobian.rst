mindscience.common.batched_jacobian
====================================

.. py:function:: mindscience.common.batched_jacobian(model)

    计算网络模型的雅可比矩阵。

    参数：
        - **model** (mindspore.nn.Cell) - 输入维度为 in_channels 输出维度为 out_channels 的网络模型。

    返回：
        Tensor，用于计算雅可比矩阵的Jacobian实例。输入维度为 :math:`[batch_size，in_channels]`，输出维度为 :math:`[out_channels，batch_size，in_channels]`。
    
    .. note::
        使用 `mindspore.jacrev` 所在的 MindSpore 版本必须 >= 2.0.0。
