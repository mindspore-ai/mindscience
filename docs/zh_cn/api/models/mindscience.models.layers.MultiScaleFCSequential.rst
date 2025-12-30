mindscience.models.layers.MultiScaleFCSequential
===================================================

.. py:class:: mindscience.models.layers.MultiScaleFCSequential(in_channels, out_channels, layers, neurons, residual=True, act='sin', weight_init='normal', weight_norm=False, has_bias=True, bias_init='default', num_scales=4, amp_factor=1.0, scale_factor=2.0, input_scale=None, input_center=None, latent_vector=None)

    多尺度全连接网络。

    参数：
        - **in_channels** (int) - 输入空间的通道数。
        - **out_channels** (int) - 输出空间的通道数。
        - **layers** (int) - 层数总数，包括输入/隐藏/输出层。
        - **neurons** (int) - 隐藏层的神经元数。
        - **residual** (bool, 可选) - 隐藏层的全连接残差块。默认值：``True``。
        - **act** (Union[str, Cell, Primitive, None], 可选) - 应用于全连接层输出的激活函数，例如 ``"ReLU"``。默认值：``"sin"``。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number], 可选) - 可训练的 weight_init 参数。数据类型与 `input` 相同。str 的取值请参考函数 `initializer`。默认值：``"normal"``。
        - **weight_norm** (bool, 可选) - 是否计算权重的平方和。默认值：``False``。
        - **has_bias** (bool, 可选) - 指定层是否使用偏置向量。默认值：``True``。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number], 可选) - 可训练的 bias_init 参数。数据类型与 `input` 相同。str 的取值请参考函数 `initializer`。默认值：``"default"``。
        - **num_scales** (int, 可选) - 多尺度网络的子网数。默认值：``4``。
        - **amp_factor** (Union[int, float], 可选) - 输入的放大系数。默认值：``1.0``。
        - **scale_factor** (Union[int, float], 可选) - 基础缩放系数。默认值：``2.0``。
        - **input_scale** (Union[list, None], 可选) - 输入 x/y/t 的缩放系数。如果不为 ``None``，将在网络中设置输入之前对输入进行缩放。默认值：``None``。
        - **input_center** (Union[list, None], 可选) - 坐标平移的中心位置。如果不为 ``None``，将在网络中设置输入之前对输入进行平移。默认值：``None``。
        - **latent_vector** (Union[Parameter, None], 可选) - 可训练参数，将与采样输入连接并在训练期间更新。默认值：``None``。

    输入：
        - **input** (Tensor) - 形状为 :math:`(*, in\_channels)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(*, out\_channels)` 的张量。

    异常：
        - **TypeError** - 如果 `num_scales` 不是整数。
        - **TypeError** - 如果 `amp_factor` 既不是整数也不是浮点数。
        - **TypeError** - 如果 `scale_factor` 既不是整数也不是浮点数。
        - **TypeError** - 如果 `latent_vector` 既不是 Parameter 也不是 ``None``。
