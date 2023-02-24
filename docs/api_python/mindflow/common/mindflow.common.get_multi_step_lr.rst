mindflow.common.get_multi_step_lr
=================================

.. py:function:: mindflow.common.get_multi_step_lr(lr_init, milestones, gamma, steps_per_epoch, last_epoch)

    epoch的数量到达其中一个milestone时，学习率按 `gamma` 进行衰减，生成学习率数组。

    根据给定的 `milestone` 和 `lr_init` 计算学习速率。对于 `milestone` 为 :math:`(M_1, M_2, ..., M_t, ..., M_N)` ， `lr_init` 为 :math:`(x_1, x_2, ..., x_t, ..., x_N)` 。N表示 `milestone` 的长度。设输出学习速度为 `y` ，则对于第 `i` 步，计算decayed_learning_rate[i]的表达式为：

    .. math::
        y[i] = x_t,\ for\ i \in [M_{t-1}, M_t)

    参数：
        - **lr_init** (float) - 初始学习速率，正值。
        - **milestones** (Union[list[int], tuple[int]]) - 学习率改变时epoch的数量，非负值。
        - **gamma** (float) - 学习速率调整倍数。
        - **steps_per_epoch** (int) - 每个epoch的步数，正值。
        - **last_epoch** (int) - 总epoch的数量，正值。

    返回：
        Numpy.array，学习率数组。

    异常：
        - **TypeError** - 如果 `lr_init` 或 `gamma` 不是float。
        - **TypeError** - 如果 `steps_per_epoch` 或 `last_epoch` 不是int。
        - **TypeError** - 如果 `milestones` 既不是tuple也不是list。