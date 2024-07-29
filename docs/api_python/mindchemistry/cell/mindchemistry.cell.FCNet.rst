mindchemistry.cell.FCNet
============================

.. py:class:: mindchemistry.cell.FCNet(channels, weight_init='normal', has_bias=True, bias_init='zeros', has_dropout=False, dropout_rate=0.5, has_layernorm=False, layernorm_epsilon=1e-7, has_activation=True, act='relu')

    一个全连接层网络。对输入数据应用一系列全连接层。

    参数：
        - **channels** (List) - 每个全连接层的通道数列表。
        - **weight_init** (Union[str, float, mindspore.common.initializer, List]) - 初始化层权重。如果 **weight_init** 是列表，则每个元素对应每个层。默认值：``'normal'``。
        - **has_bias** (Union[bool, List]) - 表示全连接层是否有偏置的开关。如果 **has_bias** 是列表，则每个元素对应每个全连接层。默认值：``True``。
        - **bias_init** (Union[str, float, mindspore.common.initializer, List]) - 全连接层偏置的初始化方法。如果 **bias_init** 是列表，则每个元素对应每个全连接层。默认值：``'zeros'``。
        - **has_dropout** (Union[bool, List]) - 线性块是否有 dropout 层的开关。如果 **has_dropout** 是列表，则每个元素对应每个层。默认值：``False``。
        - **dropout_rate** (float) - Dropout 层的丢弃率，必须是范围在 (0, 1] 的浮点数。如果 **dropout_rate** 是列表，则每个元素对应每个 dropout 层。默认值：``0.5``。
        - **has_layernorm** (Union[bool, List]) - 线性块是否有层归一化层的开关。如果 **has_layernorm** 是列表，则每个元素对应每个层。默认值：``False``。
        - **layernorm_epsilon** (float) - 层归一化层的超参数 epsilon。如果 **layernorm_epsilon** 是列表，则每个元素对应每个层归一化层。默认值：``1e-7``。
        - **has_activation** (Union[bool, List]) - 线性块是否有激活函数层的开关。如果 **has_activation** 是列表，则每个元素对应每个层。默认值：``True``。
        - **act** (Union[str, None, List]) - 线性块中的激活函数。如果 **act** 是列表，则每个元素对应每个激活函数层。默认值：``'relu'``。

    输入：
        - **input** (Tensor) - 形状为 :math:`(*, channels[0])` 的Tensor。

    输出：
        - **output** (Tensor) - 形状为 :math:`(*, channels[-1])` 的Tensor。

