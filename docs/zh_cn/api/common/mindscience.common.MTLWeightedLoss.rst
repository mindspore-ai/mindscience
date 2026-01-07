mindscience.common.MTLWeightedLoss
===================================

.. py:class:: mindscience.common.MTLWeightedLoss(num_losses, bound_param=0.0)

    自动计算基于 MTL 策略的加权多任务损失。更多信息请参考 `MTL weighted losses <https://arxiv.org/pdf/1805.06334.pdf>`_ 。

    参数：
        - **num_losses** (int) - 多任务损失的数量，应为正整数。
        - **bound_param** (float, 可选) - 当仅有的边界值高于给定常数时，用于对权重和正则项进行平方加和。默认 ``0.0`` 。

    输入：
        - **input** (tuple[Tensor]) - 输入数据。

    输出：
        - **output** (Tensor) - 多任务加权策略下的损失值。
