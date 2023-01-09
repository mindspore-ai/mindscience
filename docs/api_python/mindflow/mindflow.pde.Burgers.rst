.. py:class:: mindflow.pde.Burgers(model, loss_fn=mindspore.nn.MSELoss())

    基于PDEWithLoss定义的一维Burgers方程求解问题。

    参数：
        - **model** (mindspore.nn.Cell) - 用于训练的网络模型。
        - **loss_fn** (Union[None, mindspore.nn.Cell]) - 损失函数。默认：mindspore.nn.MSELoss。

    .. py::method:: pde()

        抽象方法，基于sympy定义的一维Burgers控制方程。

        返回：
            dict，自定义sympy符号方程。