mindscience.common.get_multi_step_lr
=====================================

.. py:function:: mindscience.common.get_multi_step_lr(lr_init, milestones, gamma, steps_per_epoch, last_epoch)

    当训练轮次达到给定 `milestones` 之一时，学习率按 `gamma` 进行衰减，生成学习率数组。

    根据给定的 `milestone` 和 `lr_init` 计算学习速率。对于 `milestone` 为 :math:`(M_1, M_2, ..., M_t, ..., M_N)` ， `lr_init` 为 :math:`(x_1, x_2, ..., x_t, ..., x_N)` 。 N 表示 `milestone` 的长度。设输出学习率为 `y` ，则对于第 i 步，计算 `decayed_learning_rate[i]` 的表达式为：

    .. math::
        y[i] = x_t,\ for\ i \in [M_{t-1}, M_t)

    参数：
        - **lr_init** (float) - 初始学习率，必须为正浮点数。
        - **milestones** (Union[list[int], tuple[int]]) - 学习率改变时训练轮数的列表或元组，其中每个元素必须大于 ``0``。
        - **gamma** (float) - 学习率衰减因子。
        - **steps_per_epoch** (int) - 每个 epoch 的 step 数，必须为正整数。
        - **last_epoch** (int) - 训练的总 epoch 数，必须为正整数。

    返回：
        Numpy.array，学习率数组。

    异常：
        - **TypeError** - 当 `lr_init` 或 `gamma` 不是 float 类型时抛出。
        - **TypeError** - 当 `steps_per_epoch` 或 `last_epoch` 不是 int 类型时抛出。
        - **TypeError** - 当 `milestones` 既不是 tuple 也不是 list 时抛出。
