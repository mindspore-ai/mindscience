mindelec.architecture.MultiScaleFCCell
======================================

.. py:class:: mindelec.architecture.MultiScaleFCCell(in_channel, out_channel, layers, neurons, residual=True, act='sin', weight_init='normal', has_bias=True, bias_init='default', num_scales=4, amp_factor=1.0, scale_factor=2.0, input_scale=None, input_center=None, latent_vector=None)

    多尺度神经网络。

    参数：
        - **in_channel** (int) - 输入空间中的通道数。
        - **out_channel** (int) - 输出空间中的通道数。
        - **layers** (int) - 层总数，包括输入/隐藏/输出层。
        - **neurons** (int) - 隐藏层的神经元数量。
        - **residual** (bool) - 隐藏层的残差块是否为全连接。默认值： ``True``。
        - **act** (Union[str, Cell, Primitive, None]) - 应用于全连接层输出的激活函数，例如 ``"ReLU"``。默认值： ``"sin"``。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始权重值。dtype与输入 `input` 相同。str的值可参考函数 `mindspore.common.initializer`。默认值： ``"normal"``。
        - **has_bias** (bool) - 指定图层是否使用偏置向量。默认值： ``True``。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始偏差值。dtype为与输入 `input` 相同。str的值可参考函数 `mindspore.common.initializer`。默认值： ``"default"``。
        - **num_scales** (int) - 多规模网络的子网号。默认值： ``4``。
        - **amp_factor** (Union[int, float]) - 输入的放大系数。默认值： ``1.0``。
        - **scale_factor** (Union[int, float]) - 基本比例因子。默认值： ``2.0``。
        - **input_scale** (Union[list, None]) - 输入x/y/t的比例因子。如果不是 ``None``，则输入将在网络中设置之前缩放。默认值： ``None``。
        - **input_center** (Union[list, None]) - 坐标转换的中心位置。如果不是 ``None``，则输入将在网络中设置之前转换。默认值： ``None``。
        - **latent_vector** (Union[Parameter, None]) - 与采样输入连接，并在训练期间更新的可训练参数。默认值： ``None``。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。

    输出：
        shape为 :math:`(*, out\_channels)` 的Tensor。

    异常：
        - **TypeError** - 如果 `num_scales` 不是int。
        - **TypeError** - 如果 `amp_factor` 既不是int也不是float。
        - **TypeError** - 如果 `scale_factor` 既不是int也不是float。
        - **TypeError** - 如果 `latent_vector` 既不是Parameter，也不是 ``None``。
