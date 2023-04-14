mindflow.cell.ResBlock
=========================

.. py:class:: mindflow.cell.ResBlock(in_channels, out_channels, weight_init="normal", bias_init="zeros", has_bias=True, activation=None, weight_norm=False)

    密集层的残差模块。

    参数：
        - **in_channels** (int) - 输入中的通道数。
        - **out_channels** (int) - 输出中的通道数。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始权重值。数据类型与输入 `input` 相同。str的值引用函数 `initializer` 。默认值： ``"normal"``。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始偏差值。数据类型与输入 `input` 相同。str的值引用函数 `initializer` 。默认值： ``"zeros"``。
        - **has_bias** (bool) - 指定图层是否使用偏置向量。默认值： ``True``。
        - **activation** (Union[str, Cell, Primitive, None]) - 应用于密集层输出的激活函数。默认值： ``None``。
        - **weight_norm** (bool) - 是否计算权重的平方和。默认值： ``False``。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。

    输出：
        shape为 :math:`(*, out\_channels)` 的Tensor。
    
    异常：
        - **ValueError** - 如果 `in_channels` 不等于 `out_channels` 。
        - **TypeError** - 如果 `activation` 类型不是str或者Cell或者Primitive。
        
