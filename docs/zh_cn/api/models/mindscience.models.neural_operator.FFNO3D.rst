mindscience.models.neural_operator.FFNO3D
===================================================

.. py:class:: mindscience.models.neural_operator.FFNO3D(in_channels, out_channels, n_modes, resolutions, hidden_channels=20, lifting_channels=None, projection_channels=128, factor=1, n_layers=4, n_ff_layers=2, ff_weight_norm=False, layer_norm=True, share_weight=False, r_padding=0, data_format='channels_last', positional_embedding=True, dft_compute_dtype=mstype.float32, ffno_compute_dtype=mstype.float16)

    3D 因子化傅里叶神经算子，通常包含一个提升层、一个因子化傅里叶块层和一个投影层。详情请参阅
    `A. Tran, A. Mathews, et. al: FACTORIZED FOURIER NEURAL OPERATORS <https://arxiv.org/pdf/2010.08895.pdf>`_。

    参数:
        - **in_channels** (int) - 输入空间的通道数。
        - **out_channels** (int) - 输出空间的通道数。
        - **n_modes** (Union[int, list(int)]) - 傅里叶层线性变换后保留的模式数。
        - **resolutions** (Union[int, list(int)]) - 输入张量的分辨率。
        - **hidden_channels** (int, 可选) - FNOBlock 输入和输出的通道数。默认值：``20``。
        - **lifting_channels** (int, 可选) - 提升层中间通道的通道数。默认值：``None``。
        - **projection_channels** (int, 可选) - 投影层中间通道的通道数。默认值：``128``。
        - **factor** (int, 可选) - 前馈神经网络隐藏层中的神经元数。默认值：``1``。
        - **n_layers** (int, 可选) - 傅里叶层嵌套的次数。默认值：``4``。
        - **n_ff_layers** (int, 可选) - 前馈神经网络中的层数（隐藏层）。默认值：``2``。
        - **ff_weight_norm** (bool, 可选) - 是否在前馈中进行权重归一化。用作保留功能接口，前馈中不支持权重归一化。默认值：``False``。
        - **layer_norm** (bool, 可选) - 是否在前馈中进行层归一化。默认值：``True``。
        - **share_weight** (bool, 可选) - SpectralConv 层之间是否共享权重。默认值：``False``。
        - **r_padding** (int, 可选) - 在某个维度上对张量右侧进行填充的数字。默认值：``0``。
        - **data_format** (str, 可选) - 输入数据通道序列。默认值：``"channels_last"``。
        - **positional_embedding** (bool, 可选) - 是否嵌入位置信息。默认值：``True``。
        - **dft_compute_dtype** (dtype.Number, 可选) - SpectralConvDft 中 DFT 的计算类型。默认值：``mstype.float32``。
        - **ffno_compute_dtype** (dtype.Number, 可选) - fno 跳跃 MLP 的计算类型。可选择 ``mstype.float32`` 或 ``mstype.float16``。GPU 后端推荐使用 ``mstype.float32``，Ascend 后端推荐使用 ``mstype.float16``。默认值：``mstype.float16``。

    输入:
        - **x** (Tensor) - 形状为 :math:`(batch\_size, resolution, in\_channels)` 的张量。

    输出:
        - **output** (Tensor) - 形状为 :math:`(batch\_size, resolution, out\_channels)` 的张量。

    异常:
        - **ValueError** - 如果 `ff_weight_norm` 不是 ``False``。
