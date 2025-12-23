mindscience.models.neural_operator.FFNOBlocks
===================================================

.. py:class:: mindscience.models.neural_operator.FFNOBlocks(in_channels, out_channels, n_modes, resolutions, factor=1, n_ff_layers=2, ff_weight_norm=False, layer_norm=True, dropout=0.0, r_padding=0, use_fork=False, forecast_ff=None, backcast_ff=None, fourier_weight=None, dft_compute_dtype=mstype.float32, ffno_compute_dtype=mstype.float16)

    FFNOBlock，通常伴随一个提升层和一个投影层，是因子化傅里叶神经算子的一部分。它包含一个因子化傅里叶层。详情请参阅
    `A. Tran, A. Mathews, et. al: FACTORIZED FOURIER NEURAL OPERATORS <https://arxiv.org/pdf/2010.08895.pdf>`_。

    参数:
        - **in_channels** (int) - 输入空间的通道数。
        - **out_channels** (int) - 输出空间的通道数。
        - **n_modes** (Union[int, list(int)]) - 傅里叶层线性变换后保留的模式数。
        - **resolutions** (Union[int, list(int)]) - 输入张量的分辨率。
        - **factor** (int, 可选) - 前馈神经网络隐藏层中的神经元数。默认值：``1``。
        - **n_ff_layers** (int, 可选) - 前馈神经网络中的层数（隐藏层）。默认值：``2``。
        - **ff_weight_norm** (bool, 可选) - 是否在前馈中进行权重归一化。用作保留功能接口，前馈中不支持权重归一化。默认值：``False``。
        - **layer_norm** (bool, 可选) - 是否在前馈中进行层归一化。默认值：``True``。
        - **dropout** (float, 可选) - 应用 dropout 正则化时被丢弃的百分比值。默认值：``0.0``。
        - **r_padding** (int, 可选) - 在某个维度上对张量右侧进行填充的数字。如果输入不是周期性的，则填充域。默认值：``0``。
        - **use_fork** (bool, 可选) - 是否执行预报。默认值：``False``。
        - **forecast_ff** (Feedforward, 可选) - 生成"回波"输出的前馈网络。默认值：``None``。
        - **backcast_ff** (Feedforward, 可选) - 生成"预报"输出的前馈网络。默认值：``None``。
        - **fourier_weight** (ParameterTuple[Parmemter], 可选) - 在频域中变换数据的傅里叶权重，包含一个长度为 2N 的 Parmemter 参数元组。

          - 偶数索引（0, 2, 4, ...）表示复数参数的实部。
          - 奇数索引（1, 3, 5, ...）表示复数参数的虚部。

          默认值：``None``，表示未提供数据。
        - **dft_compute_dtype** (dtype.Number, 可选) - SpectralConv 中 DFT 的计算类型。默认值：``mstype.float32``。
        - **ffno_compute_dtype** (dtype.Number, 可选) - ffno 跳跃 MLP 的计算类型。默认值：``mstype.float16``。可选择 ``mstype.float32`` 或 ``mstype.float16``。GPU 后端推荐使用 ``mstype.float32``，Ascend 后端推荐使用 ``mstype.float16``。

    输入:
        - **x** (Tensor) - 形状为 :math:`(batch\_size, in\_channels, resolution)` 的张量。

    输出:
        - **output** (Tensor) - 形状为 :math:`(batch\_size, out\_channels, resolution)` 的张量。

    异常:
        - **ValueError** - 如果 `ff_weight_norm` 不是 ``False``。
