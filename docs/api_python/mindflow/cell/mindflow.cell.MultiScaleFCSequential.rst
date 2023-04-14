mindflow.cell.MultiScaleFCSequential
=====================================

.. py:class:: mindflow.cell.MultiScaleFCSequential(in_channels, out_channels, layers, neurons, residual=True, act="sin", weight_init="normal", weight_norm=False, has_bias=True, bias_init='default', num_scales=4, amp_factor=1.0, scale_factor=2.0, input_scale=None, input_center=None, latent_vector=None)

    多尺度的全连接神经网络。

    参数：
        - **in_channels** (int) - 输入中的通道数。
        - **out_channels** (int) - 输出中的通道数。
        - **layers** (int) - 层总数，包括输入/隐藏/输出层。
        - **neurons** (int) - 隐藏层的神经元数量。
        - **residual** (bool) - 隐藏层的残差块的全连接。默认值： ``True``。
        - **act** (Union[str, Cell, Primitive, None]) - 激活应用于全连接层输出的函数，例如 ``"ReLU"``。默认值： ``"sin"``。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始权重值。数据类型与输入 `input` 相同。str的值引用函数 `initializer` 。默认值： ``"normal"``。
        - **weight_norm** (bool) - 是否计算权重的平方和。默认值： ``False``。
        - **has_bias** (bool) - 指定图层是否使用偏置向量。默认值： ``True``。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 可训练的初始偏差值。数据类型与输入 `input` 相同。str的值引用函数 `initializer` 。默认值： ``"default"``。
        - **num_scales** (int) - 多规模网络的子网号。默认值： ``4``。
        - **amp_factor** (Union[int, float]) - 输入的放大系数。默认值： ``1.0``。
        - **scale_factor** (Union[int, float]) - 基本比例因子。默认值： ``2.0``。
        - **input_scale** (Union[list, None]) - 输入x/y/t的比例因子。如果不是 ``None``，则输入将在网络中设置之前缩放。默认值： ``None``。
        - **input_center** (Union[list, None]) - 坐标转换的中心位置。如果不是 ``None``，则输入将在网络中设置之前翻译。默认值： ``None``。
        - **latent_vector** (Union[Parameter, None]) - 将与采样输入连接的可训练的parameter并在训练期间更新。默认值： ``None``。

    输入：
        - **input** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。

    输出：
        shape为 :math:`(*, out\_channels)` 的Tensor。
    
    异常：
        - **TypeError** - 如果 `num_scales` 不是int类型。
        - **TypeError** - 如果 `amp_factor` 不是int及或者float类型。
        - **TypeError** - 如果 `scale_factor` 不是int及或者float类型。
        - **TypeError** - 如果 `latent_vector` 不是Parameter类型或者 ``None``。
        