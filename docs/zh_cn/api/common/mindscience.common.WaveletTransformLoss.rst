mindscience.common.WaveletTransformLoss
========================================

.. py:class:: mindscience.common.WaveletTransformLoss(wave_level=2, regroup=False)

    多层小波变换损失函数。

    参数：
        - **wave_level** (int, 可选) - 小波变换级数，应为正整数。默认值： ``2``。
        - **regroup** (bool, 可选) - 小波变换损失中误差的重新组合方式。默认值： ``False``。

    输入：
        - **input (tuple(Tensor, Tensor))** - Tensor 构成的元组。Tensor 的形状为 :math:`(B*H*W/(P*P), P*P*C)` ，其中 B 表示批次大小。H、W 分别表示图像的高度和宽度。P 表示 patch 大小。C 表示特征通道。

    输出：
        Tensor，小波变换损失函数输出。
