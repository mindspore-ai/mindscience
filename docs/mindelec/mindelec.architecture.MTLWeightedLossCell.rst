mindelec.architecture.MTLWeightedLossCell
=========================================

.. py:class:: mindelec.architecture.MTLWeightedLossCell(num_losses)

    MTL策略自动加权多任务损失。

    参数：
        - **num_losses** (int) - 多任务损失的数量，应为正整数。

    输入：
        - **input** (tuple[Tensor]) - 输入数据。

    输出：
        Tensor。
