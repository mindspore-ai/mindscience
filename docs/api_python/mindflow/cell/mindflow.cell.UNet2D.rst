mindflow.cell.UNet2D
=========================

.. py:class:: mindflow.cell.UNet2D(in_channels, out_channels, base_channels, data_format="NHWC", kernel_size=2, stride=2)

    二维UNet模型。
    UNet是一个呈U型的卷积神经网络。它由一个捕捉上下文的编码器和一个实现精确定位的解码器组成。
    具体细节可以参见 `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_ 。

    参数：
        - **in_channels** (int) - 输入的输入特征维度。
        - **out_channels** (int) - 输出的输出特征维度。
        - **base_channels** (int) - UNet的基础维度，以此维度为基础，UNet先成倍增加维度，后成倍减少维度。
        - **data_format** (str) - 输入数据的数据类型。默认值： ``"NHWC"``。
        - **kernel_size** (int) - 卷积计算的卷积核大小。默认值： ``2``。
        - **stride** (Union[int, tuple[int]]) - 卷积计算的stride大小。整型表示卷积核横向和纵向均移动相同的步长，元组型由两个整数组成，分别表示横向和纵向的卷积核移动步长。默认值： ``2``。

    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, feature\_size, image\_height, image\_width)` 的Tensor。

    输出：
        - **output** (Tensor) - shape为 :math:`(batch\_size, feature\_size, image\_height, image\_width)` 的Tensor。