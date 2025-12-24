mindscience.common.unpatchify
=============================

.. py:function:: mindscience.common.unpatchify(labels, img_size=(192, 384), patch_size=16, nchw=False)

    将序列形式的图像/栅格张量恢复为图像/栅格张量。

    参数：
        - **labels** - 每个位置的输出维度。
        - **img_size** - 输入图像尺寸，默认 ``(192, 384)``。
        - **patch_size** - 图像的 patch 大小，默认 ``16``。
        - **nchw** - 若为 ``True``，则输出形状为 ``NCHW``。

    返回：
        Tensor。形状为 :math:`(N, H, W, C)`。