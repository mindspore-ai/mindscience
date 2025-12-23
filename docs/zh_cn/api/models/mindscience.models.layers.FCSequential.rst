mindscience.models.layers.FCSequential
===================================================

.. py:class:: mindscience.models.layers.FCSequential(in_channels, out_channels, layers, neurons, residual=True, act='sin', weight_init='normal', has_bias=True, bias_init='default', weight_norm=False)

    全连接层的顺序容器，全连接层按顺序添加到容器中。

    参数：
        - **in_channels** (int) - 输入空间的通道数。
        - **out_channels** (int) - 输出空间的通道数。
        - **layers** (int) - 层数总数，包括输入/隐藏/输出层。
        - **neurons** (int) - 隐藏层的神经元数。
        - **residual** (bool, 可选) - 隐藏层的全连接残差块。默认值：``True`` 。
        - **act** (Union[str, Cell, Primitive, None], 可选) - 应用于全连接层输出的激活函数，例如 ``"ReLU"`` 。默认值：``"sin"`` 。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number], 可选) - 可训练的 weight_init 参数。数据类型与输入 x 相同。str 的取值请参考函数 `initializer` 。默认值：``"normal"`` 。
        - **has_bias** (bool, 可选) - 指定层是否使用偏置向量。默认值：``True`` 。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number], 可选) - 可训练的 bias_init 参数。数据类型与输入 x 相同。str 的取值请参考函数 `initializer` 。默认值：``"default"`` 。
        - **weight_norm** (bool, 可选) - 是否计算权重的平方和。默认值：``False`` 。

    输入：
        - **input** (Tensor) - 形状为 :math:`(*, in\_channels)` 的张量。

    输出：
        形状为 :math:`(*, out\_channels)` 的张量。

    异常：
        - **TypeError** - 如果 `layers` 不是整数。
        - **TypeError** - 如果 `neurons` 不是整数。
        - **TypeError** - 如果 `residual` 不是布尔值。
        - **ValueError** - 如果 `layers` 小于 3。

