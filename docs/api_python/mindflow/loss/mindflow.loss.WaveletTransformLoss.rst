mindflow.loss.WaveletTransformLoss
==================================

.. py:class:: mindflow.loss.WaveletTransformLoss(wave_level=2, regroup=False)

    多级小波变换损失。

    参数：
        - **wave_level** (int) - 小波变换级数，应为正整数。默认值： ``2``。
        - **regroup** (bool) - 小波变换损失的regroup误差组合形式。默认值： ``False``。

    输入：
        - **input** - 张量构成的tuple。Tensor的shape为 :math:`(B*H*W/(P*P), P*P*C)` ，其中B表示批次大小。H、W分别表示图像的高度和宽度。P表示补丁大小。C表示特征通道。

    输出：
        Tensor。小波变换损失函数输出。

    异常：
        - **TypeError** - 如果 `wave_level` 不是int。
        - **TypeError** - 如果 `regroup` bool。