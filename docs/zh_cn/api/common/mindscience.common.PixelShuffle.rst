mindscience.common.PixelShuffle
================================

.. py:class:: mindscience.common.PixelShuffle(upscale_factor)

    对由多个输入平面组成的输入信号应用 pixelshuffle 操作，该操作可用于实现步长为 :math:`1/r` 的高效子像素卷积。
    详细内容可参考 `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ 。

    通常，输入的形状为 :math:`(*, C \times r^2, H, W)`，输出的形状为 :math:`(*, C, H \times r, W \times r)`，其中 `r` 表示上采样因子，`*` 表示零个或多个 batch 维度。

    参数：
        - **upscale_factor** (int) - 上采样因子，正整数。

    输入：
        - **x** (Tensor) - 形状为 :math:`(*, C \times r^2, H, W)` 的张量，维度数需大于 ``2``，且倒数第三个维度的长度必须能够被 `upscale_factor` 的平方整除。

    输出：
        - **output** (Tensor) - 形状为 :math:`(*, C, H \times r, W \times r)` 的张量。

    异常：
        - **ValueError** - 当 `upscale_factor` 不是正整数时抛出。
        - **ValueError** - 当 `x` 的倒数第三个维度长度不能被 `upscale_factor` 的平方整除时抛出。
        - **TypeError** - 当 `x` 的维度小于 ``3`` 时抛出。
