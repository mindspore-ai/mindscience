mindflow.operators.batched_hessian
==================================

.. py:function:: mindflow.operators.batched_hessian(model)

    计算网络模型的海森矩阵。

    参数：
        - **model** (mindspore.nn.Cell) - 输入维度为in_channels输出维度为out_channels的网络模型。

    返回：
        Function，用于计算海森矩阵的Hessian实例。输入维度为：[batch_size，in_channels]，输出维度为：[out_channels，in_channels，batch_size，in_channels]。

    .. note::
        要求MindSpore版本 >= 2.0.0调用如下接口： `mindspore.jacrev` 。