mindflow.pde.UnsteadyFlowWithLoss
=================================

.. py:class:: mindflow.pde.UnsteadyFlowWithLoss(model, t_in=1, t_out=1, loss_fn='mse', data_format='NTCHW')

    基于数据驱动的非定常流体问题求解的基类。

    参数：
        - **model** (mindspore.nn.Cell) - 用于训练的网络模型。
        - **t_in** (int) - 初始步长。默认值： ``1``。
        - **t_out** (int) - 输出步长。 默认值： ``1``。
        - **loss_fn** (Union[str, Cell]) - 损失函数。默认值： ``'mse'``。
        - **data_format** (str) - 数据格式。默认值： ``'NTCHW'``。

    .. py:method:: get_loss(inputs, labels)

        计算训练或测试模型的损失。

        参数：
            - **inputs** (Tensor) - 模型输入数据。
            - **labels** (Tensor) - 样本真实值。

        返回：
            float，损失值。

    .. py:method:: step(inputs)

        支持单步或多步训练。

        参数：
            - **inputs** (Tensor) - 输入数据，数据格式为'NTCHW'或'MHWTC'。

        返回：
            List(Tensor)，格式为'NTCHW'或'MHWTC'的数据。