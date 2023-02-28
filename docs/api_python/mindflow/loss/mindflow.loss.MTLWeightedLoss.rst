mindflow.loss.MTLWeightedLoss
=================================

.. py:class:: mindflow.loss.MTLWeightedLoss(num_losses, bound_param=0.0)

    计算MTL策略自动加权多任务损失。请参考 `自动加权进行多任务学习 <https://arxiv.org/pdf/1805.06334.pdf>`_ 。

    参数：
        - **num_losses** (int) - 多任务损失的数量，应为正整数。
        - **bound_param** (float) - 当边界值大于某个给定常数时，对权重和正则项的增加。

    输入：
        - **input** (tuple[Tensor]) - 输入数据。

    输出：
        Tensor。多任务学习自动加权计算出的损失。
