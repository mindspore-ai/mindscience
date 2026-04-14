mindsponge.optimizer.SteepestDescent
====================================

.. py:class:: mindsponge.optimizer.SteepestDescent(crd, learning_rate=1e-03, factor=1.001, nonh_mask=None, max_shift=1.0)

    随着学习速度的提高，最陡峭的下降(梯度下降)优化器。

    参数：
        - **crd** (tuple) - 一个参数元组，第一个元素是坐标。
        - **learning_rate** (float) - 学习率。默认值：1e-03。
        - **factor** (float)- 学习率增长因子。默认值：1.001。
        - **nonh_mask** (Tensor) - 非氢原子的mask。默认值："None"。
        - **max_shift** (float) - 每一个原子可以移动的最大步长。默认值：1.0。

    输出：
        float。 `crd` 参数中的第一个元素。