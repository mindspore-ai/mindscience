mindscience.common.unpatchify
==============================

.. py:function:: mindscience.common.unpatchify(labels, img_size=(192, 384), patch_size=16, nchw=False)

    将一系列展平的 patch 序列还原为类图像张量。

    参数：
        - **labels** (Tensor) - 包含展平后 patch 表示的张量。其形状应为 :math:`(N, num_patches, patch_size * patch_size * C)`，其中 `C` 表示输出图像的通道数。
        - **img_size** (tuple(int), 可选) - 输入图像的尺寸。默认 ``(192, 384)``。
        - **patch_size** (int, 可选) - 图像的 patch 大小。默认 ``16``。
        - **nchw** (bool, 可选) - 是否以通道优先（channel-first）的格式返回输出张量。若为 ``True``，则还原后的张量形状为 :math:`(N, C, H, W)`；若为 ``False``，则还原后的张量形状为 :math:`(N, H, W, C)`。默认 ``False``。

    返回：
        Tensor，还原后的图像张量。当 `nchw` 为 ``False`` 时，张量形状为 :math:`(N, H, W, C)`；当 `nchw` 为 ``True`` 时，张量形状为 :math:`(N, C, H, W)`。