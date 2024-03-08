mindflow.cell.UNet2D
=========================

.. py:class:: mindflow.cell.UNet2D(in_channels, out_channels, base_channels, data_format="NHWC", kernel_size=2, stride=2)

    二维UNet网络。

    参数：
        - **in_channels** (int) - 输入的输入特征维度。
        - **out_channels** (int) - 输出的输出特征维度。
        - **base_channels** (int) - UNet的基础维度，以此维度为基础，UNet先成倍增加维度，后成倍减少维度。
        - **data_format** (str) - 输入数据的数据类型。默认值： ``"NHWC"``。
        - **kernel_size** (int) - 卷积计算的卷积核大小。默认值： ``2``。
        - **stride** (int) - 卷积计算的stride大小。默认值： ``2``。

    输入：
        - **input** (Tensor) - shape为 :math:`(batch\_size, feature\_size, image\_height, image\_width)` 的Tensor。

    输出：
        - **output** (Tensor) - shape为 :math:`(batch\_size, feature\_size, image\_height, image\_width)` 的Tensor。