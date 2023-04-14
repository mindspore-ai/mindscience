mindflow.pde.SteadyFlowWithLoss
===============================

.. py:class:: mindflow.pde.SteadyFlowWithLoss(model, loss_fn='mse')

    基于数据驱动的定常流动问题求解的基类。

    参数：
        - **model** (mindspore.nn.Cell) - 用于训练或测试的网络模型。
        - **loss_fn** (Union[str, Cell]) - 损失函数。默认值： ``'mse'``。

    .. py:method:: get_loss(inputs, labels)

        计算训练或测试模型的损失。

        参数：
            - **inputs** (Tensor) - 模型输入数据。
            - **labels** (Tensor) - 样本真实值。

        返回：
            float，损失值。