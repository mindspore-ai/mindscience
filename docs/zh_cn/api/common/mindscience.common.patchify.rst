mindscience.common.patchify
============================

.. py:function:: mindscience.common.patchify(label, patch_size=16)

    将类图像张量转换为由展平 patch 组成的序列。

    参数：
        - **label** (Union[int, float]) - 每个位置的输出维度。
        - **patch_size** (int, 可选) - 图像的 patch 大小。默认 ``16``。

    返回：
        Numpy.array，重塑后的数组，形状为 ``(H, W)``。