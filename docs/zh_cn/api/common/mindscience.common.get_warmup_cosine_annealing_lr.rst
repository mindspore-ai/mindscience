mindscience.common.get_warmup_cosine_annealing_lr
==================================================

.. py:function:: mindscience.common.get_warmup_cosine_annealing_lr(lr_init, steps_per_epoch, last_epoch, warmup_epochs=0, warmup_lr_init=0.0, eta_min=1e-6)

    基于余弦函数生成衰减学习率数组。当指定预热轮数时，预热阶段的学习率采用线性递增方式进行调整。
    对于第 i 步，计算余弦衰减的学习速率 `decayed_learning_rate[i]` 的表达式为:

    .. math::
        decayed\_learning\_rate[i] = eta\_min + 0.5 * (lr\_init - eta\_min) *
        (1 + cos(\frac{current\_epoch}{last\_epoch}\pi))

    其中 :math:`current\_epoch = floor(\frac{i}{steps\_per\_epoch})` 。

    当指定预热轮数时，对于预热阶段内的第 i 步，其学习率计算公式为：

    .. math::
        warmup\_learning\_rate[i] = (lr\_init - warmup\_lr\_init) * i / warmup\_steps + warmup\_lr\_init


    参数：
        - **lr_init** (float) - 初始学习速率，正值。
        - **steps_per_epoch** (int) - 每个 epoch 包含的的步数，正值。
        - **last_epoch** (int) - 训练的总 epoch 的数量，正值。
        - **warmup_epochs** (int, 可选) - 预热阶段的 epoch 数。默认 ``0``。
        - **warmup_lr_init** (float, 可选) - 预热阶段的初始学习率。默认 ``0.0``。
        - **eta_min** (float, 可选) - 最小学习率。默认 ``1e-6``。

    返回：
        Numpy.array，学习率数组。

    异常：
        - **TypeError** - 当 `lr_init`、 `warmup_lr_init` 或 `eta_min` 不是 float 类型时抛出。
        - **TypeError** - 当 `steps_per_epoch`、`warmup_epochs` 或 `last_epoch` 不是 int 类型时抛出。
