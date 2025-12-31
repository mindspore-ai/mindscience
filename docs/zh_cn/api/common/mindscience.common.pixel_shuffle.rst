mindscience.common.pixel_shuffle
=================================

.. py:function:: mindscience.common.pixel_shuffle(x, upscale_factor)

    对由多个输入平面组成的信号应用 pixel_shuffle 操作。该操作常用于实现步幅为 :math:`1/r` 的高效子像素卷积。
    更多细节请参考 `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ 。

    通常输入 `x` 的形状为 :math:`(*, C \times r^2, H, W)`，输出形状为 :math:`(*, C, H \times r, W \times r)`，其中 `r` 是上采样倍数，`*` 表示零个或多个 batch 维度。

    参数：
        - **x** (Tensor) - 形状为 :math:`(*, C \times r^2, H, W)` 的张量。`x` 的维度数需大于 ``2``，且倒数第三个维度的长度必须能够被 `upscale_factor` 的平方整除。
        - **upscale_factor** (int) - 用于提升空间分辨率的因子，必须为正整数。

    返回：
        Tensor，形状为 :math:`(*, C, H \times r, W \times r)` 的张量。

    异常：
        - **ValueError** - 当 `upscale_factor` 不是正整数时抛出。
        - **ValueError** - 当倒数第三个维度的长度不能被 `upscale_factor` 的平方整除时抛出。
        - **TypeError** - 当 `x` 的维度少于 ``3`` 时。
