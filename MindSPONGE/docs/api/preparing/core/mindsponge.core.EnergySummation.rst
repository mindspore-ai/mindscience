mindsponge.core.EnergySummation
===============================

.. py:class:: mindsponge.core.EnergySummation(num_walker: int = 1, dim_potential: int = 1, dim_bias: int = 1)

    直接计算势能和偏置的和的网络。

    参数：
        - **num_walker** (int) - 多线并行的数量。默认值：1。
        - **dim_potential** (int) - 势能的维度。默认值：1。
        - **dim_bias** (int) - 偏置势的维度。默认值：1。

    输出：
        Tensor。能量，shape(B, 1)。