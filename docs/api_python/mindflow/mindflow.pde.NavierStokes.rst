.. py:class:: mindflow.pde.NavierStokes

    基于PDEWithLoss定义的二维NavierStokes方程求解问题。

    参数：
        - **model** (mindspore.nn.Cell) - 用于训练的网络模型。
        - **re** (float) - 无量纲参数，雷诺数是流体惯性力与粘滞力的比值。默认值：100.0。
        - **loss_fn** (Union[None, mindspore.nn.Cell]) - 损失函数。默认值：mindspore.nn.MSELoss()。

    .. py::method:: pde()

        抽象方法，基于sympy定义的二维NavierStokes控制方程。

        返回：
            dict，自定义sympy符号方程。