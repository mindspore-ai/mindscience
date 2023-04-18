mindelec.architecture.ResBlock
==============================

.. py:class:: mindelec.architecture.ResBlock(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=True, activation=None)

    密集层的ResBlock。

    参数：
        - **in_channels** (int) - 输入空间中的通道数。
        - **out_channels** (int) - 输出空间中的通道数。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的weight_init参数。dtype与输入 `input` 相同。str的值可参考函数 `initializer`。默认值： ``“normal”``。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的bias_init参数。dtype为与输入 `input` 相同。str的值可参考函数 `initializer`。默认值： ``“zeros”``。
        - **has_bias** (bool) - 指定图层是否使用偏置矢量。默认值： ``True``。
        - **activation** (Union[str, Cell, Primitive, None]) - 应用于密集层输出的激活函数。默认值： ``None``。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。

    输出：
        shape为 :math:`(*, out\_channels)` 的Tensor。
    
    异常：
        - **ValueError** - 如果 `in_channels` 不等于 `out_channels`。
        - **TypeError** - 如果 `activation` 的值不是 `[str, Cell, Primitive]` 其中之一。
