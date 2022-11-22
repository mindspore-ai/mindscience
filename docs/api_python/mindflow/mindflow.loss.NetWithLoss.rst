.. py:class:: mindflow.loss.NetWithLoss(net_without_loss, constraints, loss="l2", dataset_input_map=None, mtl_weighted_cell=None, regular_loss_cell=None)

    带损失的网络封装类。

    参数：
        - **net_without_loss** (Cell) - 无损失定义的训练网络。
        - **constraints** (Constraints) - pde问题的约束函数。
        - **loss** (Union[str, dict, Cell]) - 损失函数的名称。默认值："l2"。
        - **dataset_input_map** (dict) - 数据集的输入映射，如果输入为None，第一列将被设置为输入。默认值：None。
        - **mtl_weighted_cell** (Cell) - 基于多任务学习不确定性评估的损失加权算法。默认值：None。
        - **regular_loss_cell** (Cell) - 正则化后的损失。默认值：None。

    输入：
        - **inputs** (Tensor) - 输入是包含网络输入的可变长度参数。

    输出：
        Tensor，一个shape为 :math:`(1,)` 的标量Tensor。
