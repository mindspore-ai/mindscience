mindscience.models.layers.UNet2D
===================================================

.. py:class:: mindscience.models.layers.UNet2D(in_channels, out_channels, base_channels, n_layers=4, data_format='NHWC', kernel_size=2, stride=2, activation='relu', enable_bn=True)

    2维 U-Net 模型。

    U-Net 是用于生物医学图像分割的 U 形卷积神经网络。它具有捕获上下文的收缩路径和实现精确定位的扩展路径。详情请参阅 `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_ 。

    参数：
        - **in_channels** (int) - 输入通道数。
        - **out_channels** (int) - 输出通道数。
        - **base_channels** (int) - UNet2D 的基础通道数。
        - **n_layers** (int, 可选) - 下采样和上采样卷积的数量。默认值：``4``。
        - **data_format** (str, 可选) - 输入数据格式。默认值：``"NHWC"``。
        - **kernel_size** (int, 可选) - 指定 2D 卷积核的高度和宽度。默认值：``2``。
        - **stride** (Union[int, tuple[int]], 可选) - 核移动的距离，表示移动高度和宽度的整数，或表示高度和宽度移动的两个整数的元组。默认值：``2``。
        - **activation** (Union[str, class], 可选) - 激活函数，可以是 str 或类。默认值：``"relu"``。
        - **enable_bn** (bool, 可选) - 指定卷积中是否使用批归一化。默认值：``True``。

    输入：
        - **x** (Tensor) - 形状为 :math:`(batch\_size, resolution, resolution, channels)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(batch\_size, resolution, resolution, channels)` 的张量。

    异常：
        - **ValueError** - 如果 `data_format` 不是 ``'NHWC'`` 或 ``'NCHW'``。
        - **ValueError** - 如果 `n_layers` 为 ``0``。
