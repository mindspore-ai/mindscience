mindelec.architecture.MTLWeightedLossCell
=========================================

.. py:class:: mindelec.architecture.MTLWeightedLossCell(num_losses)

    MTL策略自动加权多任务损失。请参考 `自动加权进行多任务学习 <https://arxiv.org/pdf/1805.06334.pdf>`_ 。

    参数：
        - **num_losses** (int) - 多任务损失的数量，应为正整数。

    输入：
        - **input** (tuple[Tensor]) - 输入数据。

    输出：
        Scalar。多任务学习自动加权计算出的损失。
