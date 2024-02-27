mindflow.common.get_poly_lr
===========================

.. py:function:: mindflow.common.get_poly_lr(global_step, lr_init, lr_end, lr_max, warmup_steps, total_steps, poly_power)

    生成指数衰减的学习率。学习率随着训练步数进行指数衰减。当step小于warmup_steps时，:math:`lr = step * (lr\_max - lr\_init)/warmup\_steps` ，之后 :math:`lr = lr\_end + (lr\_max - lr\_end) * [(1 - i + step)/(total\_steps - warmup\_steps)]**poly\_power`。

    参数：
        - **global_step** (int) - 当前步骤编号，非负值。
        - **lr_init** (float) - 初始学习速率，正值。
        - **lr_end** (float) - 结束学习速率，非负值。
        - **lr_max** (float) - 最大学习速率，正值。
        - **warmup_steps** (int) - 热身epoch的数量，非负值。
        - **total_steps** (int) - 训练的总epoch数量，正值。
        - **poly_power** (float) - 多学习速率的次方数，正值。

    返回：
        Numpy.array，学习率数组。
