mindelec.architecture.FCSequential
===================================

.. py:class:: mindelec.architecture.FCSequential(in_channel, out_channel, layers, neurons, residual=True, act='sin', weight_init='normal', has_bias=True, bias_init='default')

    全连接层的一个时序容器，按序放入全连接层。

    参数：
        - **in_channel** (int) - 输入中的通道数。
        - **out_channel** (int) - 输出中的通道数。
        - **layers** (int) - 层的总数，包括输入/隐藏/输出层。
        - **neurons** (int) - 隐藏层的神经元数量。
        - **residual** (bool) - 隐藏层的残差块的全连接。默认值： ``True``。
        - **act** (Union[str, Cell, Primitive, None]) - 激活应用于全连接层输出的函数，例如 ``"ReLU"``、 ``"Softmax"`` 和 ``"Tanh"`` 等。默认值： ``"sin"``。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始权重值。数据类型与输入 `input` 相同。str的值可参考函数 `mindspore.common.initializer <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html>`_ 。默认值： ``"normal"``。
        - **has_bias** (bool) - 指定图层是否使用偏置向量。默认值： ``True``。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始偏差值。数据类型与输入 `input` 相同。str的值可参考函数 `mindspore.common.initializer <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html>`_ 。默认值： ``"default"``。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。

    输出：
        shape为 :math:`(*, out\_channels)` 的Tensor。

    异常：
        - **TypeError** - 如果 `layers` 不是int。
        - **TypeError** - 如果 `neurons` 不是int。
        - **TypeError** - 如果 `residual` 不是bool值。
        - **ValueError** - 如果 `layers` 小于3。
