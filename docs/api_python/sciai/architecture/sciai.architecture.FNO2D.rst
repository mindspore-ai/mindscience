sciai.architecture.FNO2D
=========================

.. py:class:: sciai.architecture.FNO2D(in_channels, out_channels, resolution, modes, channels=20, depths=4, mlp_ratio=4, dtype=ms.float32)

    二维傅里叶神经算子（FNO2D）包含一个提升层、多个傅里叶层和一个解码器层。
    有关更多详细信息，请参考论文 `Fourier Neural Operator for Parametric Partial Differential Equations <https://arxiv.org/pdf/2010.08895.pdf>`_ 。

    参数：
        - **in_channels** (int) - 输入中的通道数。
        - **out_channels** (int) - 输出中的通道数。
        - **resolution** (int) - 输入的分辨率。
        - **modes** (int) - 要保留的低频分量的数量。
        - **channels** (int) - 输入提升尺寸后的通道数。默认值： `20`。
        - **depths** (int) - FNO层的数量。默认值： `4`。
        - **mlp_ratio** (int) - 解码器层的通道数提升比率。默认值： `4`。
        - **dtype** (dtype.Number) - 密集的计算类型。默认值： `ms.float32`。支持以下数据类型： `ms.float16` 或 `ms.float32`。GPU后端建议使用float32，Ascend后端建议使用float16。

    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, resolution, resolution, in\_channels)` 的Tensor。

    输出：
        Tensor，此FNO网络的输出。
        
        - **output** (Tensor) - shape为 :math:`(batch\_size, resolution, resolution, out\_channels)` 的Tensor。

    异常：
        - **TypeError** - 如果 `in_channels` 不是int。
        - **TypeError** - 如果 `out_channels` 不是int。
        - **TypeError** - 如果 `resolution` 不是int。
        - **TypeError** - 如果 `modes` 不是int。
        - **ValueError** - 如果 `modes` 小于1。
