mindscience.common.glorot_uniform
================================

.. py:function:: mindscience.common.glorot_uniform(fan_in, fan_out, weight_shape)

    Glorot 均匀初始化，用于生成权重张量。

    参数：
        - **fan_in** (int) - 输入特征数。
        - **fan_out** (int) - 输出特征数。
        - **weight_shape** (tuple) - 权重形状。

    返回：
        - numpy.ndarray - 生成的权重。