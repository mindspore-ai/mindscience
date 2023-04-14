mindflow.pde.FlowWithLoss
==========================

.. py:class:: mindflow.pde.FlowWithLoss(model, loss_fn='mse')

    基于数据驱动的流体问题求解的基类。

    参数：
        - **model** (mindspore.nn.Cell) - 用于训练或测试的网络模型。
        - **loss_fn** (Union[str, Cell]) - 损失函数。默认值： ``'mse'``。

    异常：
        - **TypeError** - 如果 `mode` 或 `loss_fn` 的类型不是mindspore.nn.Cell。
        - **NotImplementedError** - 如果成员函数 `get_loss` 未定义。

    .. py:method:: get_loss(inputs, labels)

        计算训练或测试模型的损失。

        参数：
            - **inputs** (Tensor) - 网络模型的输入数据。
            - **labels** (Tensor) - 样本的真值。