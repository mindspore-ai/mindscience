mindearth.core.get_warmup_cosine_annealing_lr
==============================================

.. py:function:: mindearth.core.get_warmup_cosine_annealing_lr(lr_init, steps_per_epoch, last_epoch, warmup_epochs=0, warmup_lr_init=0.0, eta_min=1e-6)

    基于余弦衰减函数计算学习率。如果指定了 `warmup epoch` ，那么 `warmup epoch` 将通过线性退火进行预热。
    对于第i步，余弦decayed_learning_rate[i]的计算公式为：

    .. math::
        decayed\_learning\_rate[i] = eta\_min + 0.5 * (lr\_init - eta\_min) * (1 + cos(\frac{current\_epoch}{last\_epoch}\pi))

    其中 :math:`current\_epoch = floor(\frac{i}{steps\_per\_epoch})`。

    如果指定了 `warmup epoch` ，则对于 `waramup epoch` 中的第i步，warmup_learning_rate[i]的计算公式为：

    .. math::
        warmup\_learning\_rate[i] = (lr\_init - warmup\_lr\_init) * i / warmup\_steps + warmup\_lr\_init

    参数：
        - **lr_init** (float) - 初始学习率，正浮点值。
        - **steps_per_epoch** (int) - 每一轮迭代的训练步数，正整数。
        - **last_epoch** (int) - 每个epoch的步数，正整数。
        - **warmup_epochs** (int) - 预热总轮数，默认： ``0`` 。
        - **warmup_lr_init** (float) - 预热初始化学习率，默认： ``0.0`` 。
        - **eta_min** (float) - 最小学习率，默认： ``1e-6`` 。

    返回：
        numpy.array，学习率数组。

    异常：
        - **TypeError** - 如果 `lr_init` 、 `warmup_lr_init` 或 `eta_min` 不是float。
        - **TypeError** - 如果 `steps_per_epoch` 、 `warmup_epochs` 或 `last_epoch` 不是int。
