mindscience.common.get_poly_lr
===============================

.. py:function:: mindscience.common.get_poly_lr(global_step, lr_init, lr_end, lr_max, warmup_steps, total_steps, poly_power)

    生成指数衰减的学习率数组。学习率随着训练步数进行指数衰减。当 step 小于 warmup_steps 时，:math:`lr = step * (lr\_max - lr\_init)/warmup\_steps` ，之后 :math:`lr = lr\_end + (lr\_max - lr\_end) * [(1 - i + step)/(total\_steps - warmup\_steps)]**poly\_power`。

    参数：
        - **global_step** (int) - 当前训练步数，非负值。
        - **lr_init** (float) - 初始学习率，正值。
        - **lr_end** (float) - 训练结束时的学习率，非负值。
        - **lr_max** (float) - 最大学习率，正值。
        - **warmup_steps** (int) - 预热阶段的轮数，非负值。
        - **total_steps** (int) - 训练总轮数，正值。
        - **poly_power** (float) - 多项式衰减的指数，正值。

   返回：
        Numpy.array，学习率数组。
