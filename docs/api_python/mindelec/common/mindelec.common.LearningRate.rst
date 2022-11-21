mindelec.common.LearningRate
============================

.. py:class:: mindelec.common.LearningRate(learning_rate, end_learning_rate, warmup_steps, decay_steps, power)

    构建学习率，包括预热学习率和衰减学习率。热身步骤大于0时，返回预热学习率，否则返回衰减学习率。

    参数：
        - **learning_rate** (float) - 基本学习速率，正数。
        - **end_learning_rate** (float) - 结束学习速率，非负数。
        - **warmup_steps** (int) - 热身步骤，非负数。
        - **decay_steps** (int) - 衰变步数，用于计算衰变学习率，正数。
        - **power** (float) - 衰变次方，用于计算衰减学习率，正数。

    输入：
       - **global_step** (Tensor) - 具有 :math:`()` 形状的当前步骤数。

    返回：
        Tensor。具有 :math:`()` 形状的当前步骤的学习速率值。
