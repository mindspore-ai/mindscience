mindscience.common.pixel_unshuffle
==================================

.. py:function:: mindscience.common.pixel_unshuffle(x, downscale_factor)

    对由多个输入平面组成的信号应用 pixel_unshuffle 操作，详细背景可以参考
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ 。

    通常输入张量形状为 :math:`(*, C, H \times r, W \times r)`，输出形状为 :math:`(*, C \times r^2, H, W)`，
    其中 `r` 是下采样倍数，`*` 代表 0 个或多个 batch 维度。

    参数：
        - **x** - 张量，形状为 :math:`(*, C, H \times r, W \times r)`。其维度必须大于 2，且倒数第二或最后一个维度的长度应能被 `downscale_factor` 整除。
        - **downscale_factor** - 下采样倍数，正整数。

    返回：
        张量，形状为 :math:`(*, C \times r^2, H, W)`。

    异常：
        - **ValueError** - 当 `downscale_factor` 不是正整数时。
        - **ValueError** - 当倒数第二或最后一个维度的长度不能被 `downscale_factor` 整除时。
        - **TypeError** - 当 `x` 的维度少于 3 时。
