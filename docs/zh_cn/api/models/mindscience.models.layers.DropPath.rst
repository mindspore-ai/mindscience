mindscience.models.layers.DropPath
===================================================

.. py:class:: mindscience.models.layers.DropPath(dropout_rate=0.0)

    按样本进行路径丢弃（随机深度）（当应用于残差块的主路径时）。

    参数：
        - **dropout_rate** (float, 可选) - 路径丢弃率，大于0且小于等于1。默认值：``0.0``。

    输入：
        - **x** (Tensor) - 输入张量。

    输出：
        - **output** (Tensor) - 输出张量，在训练期间应用了丢弃路径，在推理期间为原始输入。
