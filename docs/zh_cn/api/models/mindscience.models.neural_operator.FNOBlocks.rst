mindscience.models.neural_operator.FNOBlocks
===================================================

.. py:class:: mindscience.models.neural_operator.FNOBlocks(in_channels, out_channels, n_modes, resolutions, act='gelu', add_residual=False, dft_compute_dtype=mstype.float32, fno_compute_dtype=mstype.float16)

    FNOBlock，通常伴随一个提升层和一个投影层，是傅里叶神经算子的一部分。它包含一个傅里叶层和一个 FNO 跳跃层。
    详情请参阅 `Zongyi Li, et. al: FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS <https://arxiv.org/pdf/2010.08895.pdf>`_。

    参数：
        - **in_channels** (int) - 输入空间的通道数。
        - **out_channels** (int) - 输出空间的通道数。
        - **n_modes** (Union[int, list(int)]) - 傅里叶层线性变换后保留的模式数。
        - **resolutions** (Union[int, list(int)]) - 输入张量的分辨率。
        - **act** (Union[str, class], 可选) - 激活函数，可以是字符串或类。默认值：``"gelu"``。
        - **add_residual** (bool, 可选) - 是否在 FNOBlock 中添加残差。默认值：``False``。
        - **dft_compute_dtype** (dtype.Number, 可选) - SpectralConvDft 中 DFT 的计算类型。默认值：``mstype.float32``。
        - **fno_compute_dtype** (dtype.Number, 可选) - fno 跳跃 MLP 的计算类型。可选择 ``mstype.float32`` 或 ``mstype.float16``。GPU 后端推荐使用 ``mstype.float32``，Ascend 后端推荐使用 ``mstype.float16``。默认值：``mstype.float16``。

    输入：
        - **x** (Tensor) - 形状为 :math:`(batch\_size, in\_channels, resolution)` 的张量。

    输出：
        - **output** (Tensor) - 形状为 :math:`(batch\_size, out\_channels, resolution)` 的张量。

    异常：
        - **ValueError** - 如果 `n_modes` 的维度与 `resolutions` 的维度不相等。
