mindchemistry.cell.MLPNet
============================

.. py:class:: mindchemistry.cell.MLPNet(in_channels, out_channels, layers, neurons, weight_init='normal', has_bias=True, bias_init='zeros', has_dropout=False, dropout_rate=0.5, has_layernorm=False, layernorm_epsilon=1e-7, has_activation=True, act='relu')

    MLPNet网络。对输入数据应用一系列全连接层，其中隐藏层具有相同数量的通道。

    参数：
        - **in_channels** (int) - 输入层的通道数。
        - **out_channels** (int) - 输出层的通道数。
        - **layers** (int) - 层数。
        - **neurons** (int) - 隐藏层的通道数。
        - **weight_init** (Union[str, float, mindspore.common.initializer, List]) - 初始化层权重的方法。如果 **weight_init** 是列表，则每个元素对应每个层。默认值：``'normal'``。
        - **has_bias** (Union[bool, List]) - 指示全连接层是否有偏置的开关。如果 **has_bias** 是列表，则每个元素对应每个全连接层。默认值：``True``。
        - **bias_init** (Union[str, float, mindspore.common.initializer, List]) - 全连接层偏置的初始化方法。如果 **bias_init** 是列表，则每个元素对应每个全连接层。默认值：``'zeros'``。
        - **has_dropout** (Union[bool, List]) - 指示线性块是否有 dropout 层的开关。如果 **has_dropout** 是列表，则每个元素对应每个层。默认值：``False``。
        - **dropout_rate** (float) - Dropout 层的丢弃率，必须是范围在 (0, 1] 的浮点数。如果 **dropout_rate** 是列表，则每个元素对应每个 dropout 层。默认值：``0.5``。
        - **has_layernorm** (Union[bool, List]) - 指示线性块是否有层归一化层的开关。如果 **has_layernorm** 是列表，则每个元素对应每个层。默认值：``False``。
        - **layernorm_epsilon** (float) - 层归一化层的超参数 epsilon。如果 **layernorm_epsilon** 是列表，则每个元素对应每个层归一化层。默认值：``1e-7``。
        - **has_activation** (Union[bool, List]) - 指示线性块是否有激活函数层的开关。如果 **has_activation** 是列表，则每个元素对应每个层。默认值：``True``。
        - **act** (Union[str, None, List]) - 线性块中的激活函数。如果 **act** 是列表，则每个元素对应每个激活函数层。默认值：``'relu'``。

    输入：
        - **input** (Tensor) - 形状为 :math:`(*, channels[0])` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(*, channels[-1])` 的张量。
