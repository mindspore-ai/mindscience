mindscience.common.PixelUnshuffle
==================================

.. py:class:: mindscience.common.PixelUnshuffle(downscale_factor)

    对由多个输入平面组成的信号应用 pixelunshuffle 操作，详情可参考
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ 。

    通常输入张量形状为 :math:`(*, C, H \times r, W \times r)`，输出形状为 :math:`(*, C \times r^2, H, W)`，
    其中 `r` 是下采样倍数，`*` 表示零个或多个 batch 维度。

    参数：
        - **downscale_factor** (int) - 下采样因子，正整数。

    输入：
        - **x** (Tensor) - 形状为 :math:`(*, C, H \times r, W \times r)` 的张量，维度数需大于 ``2``，且倒数第二个维度和最后一个维度的长度必须能够被 `downscale_factor` 整除。

    输出：
        - **output** (Tensor) - 形状为 :math:`(*, C \times r^2, H, W)` 的张量。

    异常：
        - **ValueError** - 当 `downscale_factor` 不是正整数时抛出。
        - **ValueError** - 当倒数第二个维度或最后一个维度的长度不能被 `downscale_factor` 整除时抛出。
        - **TypeError** - 当 `x` 的维度小于 ``3`` 时抛出。
