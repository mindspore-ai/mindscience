mindelec.loss.NetWithLoss
=========================

.. py:class:: mindelec.loss.NetWithLoss(net_without_loss, constraints, loss='l2', dataset_input_map=None, mtl_weighted_cell=None, latent_vector=None, latent_reg=0.01)

    带损失的网络封装类。

    参数：
        - **net_without_loss** (Cell) - 无损失定义的训练网络。
        - **constraints** (Constraints) - pde问题的约束函数。
        - **loss** (Union[str, dict, Cell]) - 损失函数的名称，例如 ``"l1"``、 ``"l2"`` 和 ``"mae"`` 等。默认值： ``"l2"``。
        - **dataset_input_map** (dict) - 数据集的输入映射，如果输入为 ``None``，第一列将被设置为输入。默认值： ``None``。
        - **mtl_weighted_cell** (Cell) - 基于多任务学习不确定性评估的损失加权算法。默认值： ``None``。
        - **latent_vector** (Parameter) - 参数的张量。用于编码变分参数的控制方程的潜向量。它将与采样数据连接在一起，作为最终网络输入。默认值： ``None``。
        - **latent_reg** (float) - 潜在向量的正则化系数。默认值： ``0.01``。

    输入：
        - **inputs** (Tensor) - 输入是包含网络输入的可变长度参数。

    输出：
        Tensor，一个shape为 :math:`(1,)` 的标量Tensor。
