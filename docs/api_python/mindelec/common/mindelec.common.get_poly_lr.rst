mindelec.common.get_poly_lr
===========================

.. py:function:: mindelec.common.get_poly_lr(global_step, lr_init, lr_end, lr_max, warmup_steps, total_steps, poly_power)

    生成指数衰减学习率数组。学习率随着训练步数进行指数衰减。

    参数：
        - **global_step** (int) - 当前步骤，非负值。
        - **lr_init** (float) - 初始学习速率，正值。
        - **lr_end** (float) - 结束学习速率，非负值。
        - **lr_max** (float) - 最大学习速率，正值。
        - **warmup_steps** (int) - warmup epoch的数量，非负值。
        - **total_steps** (int) - 训练的总epoch数量，正值。
        - **poly_power** (float) - 多学习速率的次方数，正值。

    返回：
        Numpy.array，学习率数组。
