mindflow.cell.LinearBlock
=========================

.. py:class:: mindflow.cell.LinearBlock(in_channels, out_channels, weight_init="normal", bias_init="zeros", has_bias=True, activation=None)

    全连接模块。

    参数：
        - **in_channels** (int) - 输入的通道数。
        - **out_channels** (int) - 输出的通道数。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练权重的初始值。数据类型与输入 `input` 相同。str的值引用函数 `mindspore.common.initializer` 。默认值： ``"normal"``。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练偏差的初始值。数据类型与输入 `input` 相同。str的值引用函数 `mindspore.common.initializer` 。默认值： ``"zeros"``。
        - **has_bias** (bool) - 指定图层是否使用偏置向量。默认值： ``True``。
        - **activation** (Union[str, Cell, Primitive, None]) - 应用于全连接输出的激活函数层。默认值： ``None``。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。

    输出：
        shape为 :math:`(*, out\_channels)` 的Tensor。
