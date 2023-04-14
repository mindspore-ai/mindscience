mindflow.cell.FCSequential
============================

.. py:class:: mindflow.cell.FCSequential(in_channels, out_channels, layers, neurons, residual=True, act="sin", weight_init="normal", has_bias=True, bias_init="default", weight_norm=False)

    一个全连接层的顺序容器，按序放入全连接层。

    参数：
        - **in_channels** (int) - 输入中的通道数。
        - **out_channels** (int) - 输出中的通道数。
        - **layers** (int) - 层的总数，包括输入/隐藏/输出层。
        - **neurons** (int) - 隐藏层的神经元数量。
        - **residual** (bool) - 隐藏层是否使用残差网络模块。若为 ``True``，使用残差网络模块。若为 ``False``，使用线性模块。默认值： ``True``。
        - **act** (Union[str, Cell, Primitive, None]) - 激活应用于全连接层输出的函数，例如 ``"ReLU"``。默认值： ``"sin"``。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始权重值。数据类型与输入 `input` 相同。str的值引用函数 `initializer` 。默认值： ``'normal'``。
        - **has_bias** (bool) - 指定图层是否使用偏置向量。默认值： ``True``。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始偏差值。数据类型与输入 `input` 相同。str的值引用函数 `initializer` 。默认值： ``'default'``。
        - **weight_norm** (bool) - 是否计算权重的平方和。默认值： ``False``。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。

    输出：
        shape为 :math:`(*, out\_channels)` 的Tensor。
    
    异常：
        - **TypeError** - 如果 `layers` 不是int类型。
        - **TypeError** - 如果 `neurons` 不是int类型。
        - **TypeError** - 如果 `residual` 不是bool类型。
        - **ValueError** - 如果 `layers` 小于3。

