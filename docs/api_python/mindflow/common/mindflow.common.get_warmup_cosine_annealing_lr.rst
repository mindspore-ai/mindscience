mindflow.common.get_warmup_cosine_annealing_lr
==============================================

.. py:function:: mindflow.common.get_warmup_cosine_annealing_lr(lr_init, steps_per_epoch, last_epoch, warmup_epochs=0, warmup_lr_init=0.0, eta_min=1e-6)

    基于余弦函数生成衰减学习率数组。如果指定了预热epoch，将通过线性方法对预热epoch进行预热。
    对于第 `i` 步，计算余弦衰减的学习速率decayed_learning_rate[i]的表达式为:

    .. math::
        decayed\_learning\_rate[i] = eta\_min + 0.5 * (lr\_init - eta\_min) *
        (1 + cos(\frac{current\_epoch}{last\_epoch}\pi))

    其中 :math:`current\_epoch = floor(\frac{i}{steps\_per\_epoch})` .

    如果指定了预热epoch，对于预热epoch的第 `i` 步，预热学习速率的计算表达式warmup_learning_rate[i]为：

    .. math::
        warmup\_learning\_rate[i] = (lr\_init - warmup\_lr\_init) * i / warmup\_steps + warmup\_lr\_init

    参数：
        - **lr_init** (float) - 初始学习速率，正值。
        - **steps_per_epoch** (int) - 每个epoch的步数，正值。
        - **last_epoch** (int) - 总epoch的数量，正值。
        - **warmup_epochs** (int) - 热身epoch的数量，默认： ``0``。
        - **warmup_lr_init** (float) - 热身初始学习速率，默认： ``0.0``。
        - **eta_min** (float) - 学习速率最小值，默认： ``1e-6``。

    返回：
        Numpy.array，学习率数组。

    异常：
        - **TypeError** - 如果 `lr_init` 或 `warmup_lr_init` 不是float。
        - **TypeError** - 如果 `steps_per_epoch` 、 `warmup_epochs` 或 `last_epoch` 不是int。