mindelec.loss.NetWithEval
=========================

.. py:class:: mindelec.loss.NetWithEval(net_without_loss, constraints, loss='l2', dataset_input_map=None)

    具有评估损失的网络封装类。

    参数：
        - **net_without_loss** (Cell) - 无损失定义的训练网络。
        - **constraints** (Constraints) - pde问题的约束函数。
        - **loss** (Union[str, dict, Cell]) - 损失函数的名称，例如 ``"l1"``、 ``"l2"`` 和 ``"mae"`` 等。默认值： ``"l2"``。
        - **dataset_input_map** (dict) - 数据集的输入映射，如果输入为 ``None``，第一列将被设置为输入。默认值： ``None``。

    输入：
        - **inputs** (Tensor) - 输入是可变长度参数，包含网络输入和标签。

    输出：
        Tuple，包含标量损失Tensor、shape为 :math:`(N, \ldots)` 的网络输出Tensor和shape为 :math:`(N, \ldots)` 的标签Tensor。
