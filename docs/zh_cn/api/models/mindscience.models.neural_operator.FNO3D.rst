mindscience.models.neural_operator.FNO3D
===================================================

.. py:class:: mindscience.models.neural_operator.FNO3D(in_channels, out_channels, n_modes, resolutions, hidden_channels=20, lifting_channels=None, projection_channels=128, n_layers=4, data_format='channels_last', fnoblock_act='gelu', mlp_act='gelu', add_residual=False, positional_embedding=True, dft_compute_dtype=mstype.float32, fno_compute_dtype=mstype.float16)

    3D 傅里叶神经算子，通常包含一个提升层、一个傅里叶块层和一个投影层。详情请参阅
    `Zongyi Li, et. al: FOURIER NEURAL OPERATOR FOR PARAMETRIC PARTIAL DIFFERENTIAL EQUATIONS
    <https://arxiv.org/pdf/2010.08895.pdf>`_。

    参数:
        - **in_channels** (int) - 输入空间的通道数。
        - **out_channels** (int) - 输出空间的通道数。
        - **n_modes** (Union[int, list(int)]) - 傅里叶层线性变换后保留的模式数。
        - **resolutions** (Union[int, list(int)]) - 输入张量的分辨率。
        - **hidden_channels** (int, 可选) - FNOBlock 输入和输出的通道数。默认值：``20``。
        - **lifting_channels** (int, 可选) - 提升层中间通道的通道数。默认值：``None``。
        - **projection_channels** (int, 可选) - 投影层中间通道的通道数。默认值：``128``。
        - **n_layers** (int, 可选) - 傅里叶层嵌套的次数。默认值：``4``。
        - **data_format** (str, 可选) - 输入数据通道序列。支持值: ``"channels_last"`` 和 ``"channels_first"``。默认值：``"channels_last"``。
        - **fnoblock_act** (Union[str, class], 可选) - FNOBlock 的激活函数，可以是字符串或类。默认值：``"gelu"``。
        - **mlp_act** (Union[str, class], 可选) - MLP 层的激活函数，可以是字符串或类。默认值：``"gelu"``。
        - **add_residual** (bool, 可选) - 是否在 FNOBlock 中添加残差。默认值：``False``。
        - **positional_embedding** (bool, 可选) - 是否嵌入位置信息。默认值：``True``。
        - **dft_compute_dtype** (dtype.Number, 可选) - SpectralConvDft 中 DFT 的计算类型。默认值：``mstype.float32``。
        - **fno_compute_dtype** (dtype.Number, 可选) - fno 跳跃 MLP 的计算类型。可选择 ``mstype.float32`` 或 ``mstype.float16``。GPU 后端推荐使用 ``mstype.float32``，Ascend 后端推荐使用 ``mstype.float16``。默认值：``mstype.float16``。

    输入:
        - **x** (Tensor) - 形状为 :math:`(batch\_size, resolution[0], resolution[1], resolution[2], in\_channels)` 的张量。

    输出:
        - **output** (Tensor) - 形状为 :math:`(batch\_size, resolution[0], resolution[1], resolution[2], out\_channels)` 的张量。

    异常:
        - **ValueError** - 如果 `n_modes` 的维度不等于 ``3``。
        - **ValueError** - 如果 `resolutions` 的维度不等于 ``3``。