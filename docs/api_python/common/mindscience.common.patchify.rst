mindscience.common.patchify
===========================

.. py:function:: mindscience.common.patchify(label, patch_size=16)

    将图像/栅格张量按 patch 大小切分并展平成序列。

    参数：
        - **label** - 每个位置的输出维度。
        - **patch_size** - 图像的 patch 大小，默认为 ``16``。

    返回：
        重塑后的数组，形状为 ``(H, W)``。