mindscience.common.unpatchify
==============================

.. py:function:: mindscience.common.unpatchify(labels, img_size=(192, 384), patch_size=16, nchw=False)

    将一系列展平的 patch 序列还原为类图像张量。

    参数：
        - **labels** (Union[int, float]) - 每个位置的输出维度。
        - **img_size** (tuple(int), 可选) - 输入图像的尺寸。默认 ``(192, 384)``。
        - **patch_size** (int, 可选) - 图像的 patch 大小。默认 ``16``。
        - **nchw** (bool, 可选) - 若为 ``True``，则还原后的张量形状包含 ``N, C, H, W``。

    返回：
        Tensor, 形状为 :math:`(N, H, W, C)` 的张量。