mindflow.pde.NavierStokes
=========================

.. py:class:: mindflow.pde.NavierStokes(model, re=100.0, loss_fn="mse")

    基于PDEWithLoss定义的二维NavierStokes方程求解问题。

    参数：
        - **model** (mindspore.nn.Cell) - 用于训练的网络模型。
        - **re** (float) - 雷诺数（流体惯性力与粘滞力的比值）。默认值： ``100.0``。
        - **loss_fn** (Union[str, Cell]) - 损失函数。默认值： ``"mse"``。

    .. py:method:: pde()

        抽象方法，基于sympy定义的二维NavierStokes控制方程。

        返回：
            dict，自定义sympy符号方程。