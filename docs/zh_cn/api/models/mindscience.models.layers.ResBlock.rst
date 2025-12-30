mindscience.models.layers.ResBlock
===================================================

.. py:class:: mindscience.models.layers.ResBlock(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=True, activation=None, weight_norm=False)

    全连接层的 ResBlock。

    参数：
        - **in_channels** (int) - 输入空间的通道数。
        - **out_channels** (int) - 输出空间的通道数。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的 weight_init 参数。数据类型与输入 x 相同。str 的取值请参考函数 `initializer`。默认值：``"normal"``。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的 bias_init 参数。数据类型与输入 x 相同。str 的取值请参考函数 `initializer`。默认值：``"zeros"``。
        - **has_bias** (bool) - 指定层是否使用偏置向量。默认值：``True``。
        - **activation** (Union[str, Cell, Primitive, None]) - 应用于全连接层输出的激活函数。默认值：``None``。
        - **weight_norm** (bool) - 是否计算权重的平方和。默认值：``False``。

    输入：
        - **input** (Tensor) - 形状为 :math:`(*, in\_channels)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(*, out\_channels)` 的张量。

    异常：
        - **ValueError** - 如果 `in_channels` 不等于 out_channels。
        - **TypeError** - 如果 `activation` 不是 str、Cell 或 Primitive。
