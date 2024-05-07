mindflow.cell.FNO3D
=========================

.. py:class:: mindflow.cell.FNO3D(in_channels, out_channels, n_modes, resolutions, hidden_channels=20, lifting_channels=None, projection_channels=128, n_layers=4, data_format="channels_last", fnoblock_act="gelu", mlp_act="gelu", add_residual=False, positional_embedding=True, dft_compute_dtype=mstype.float32, fno_compute_dtype=mstype.float16)

    三维傅里叶神经算子（FNO3D）包含一个提升层、多个傅里叶层和一个解码器层。
    有关更多详细信息，请参考论文 `Fourier Neural Operator for Parametric Partial Differential Equations <https://arxiv.org/pdf/2010.08895.pdf>`_ 。

    参数：
        - **in_channels** (int) - 输入中的通道数。
        - **out_channels** (int) - 输出中的通道数。
        - **n_modes** (Union[int, list(int)]) - 傅里叶层中，线性变换后保留的模态数。支持整型或由三个整型数组成的列表。
        - **resolutions** (Union[int, list(int)]) - 输入中的维度。支持整型或由三个整型数组成的列表。
        - **hidden_channels** (int) - FNOBlock的输入输出通道数。默认值： ``20``。
        - **lifting_channels** (int) - 提升层中的中间层的通道数。默认值： None。
        - **projection_channels** (int) - 解码器层中的中间层的通道数。默认值： ``128``。
        - **n_layers** (int) - 傅里叶层的嵌套层数。默认值： ``4``。
        - **data_format** (str) - 输入中的数据排布顺序。默认值： ``channels_last``。支持以下类型： ``"channels_last"`` 和 ``"channels_first"`` 。
        - **fnoblock_act** (Union[str, class]) - FNOBlock层的激活函数，支持字符串或激活函数类。默认值： ``"gelu"``。
        - **mlp_act** (Union[str, class]) - MLP层的激活函数，支持字符串或激活函数类。默认值： ``gelu``。
        - **add_residual** (bool) - 是否在FNOBlock层加上残差。默认值： ``False``。
        - **positional_embedding** (bool) - 是否嵌入位置信息。默认值： ``True``。
        - **dft_compute_dtype** (dtype.Number) - SpectralConvDft层中DFT的计算类型。默认值： ``mindspore.common.dtype.float32``。支持以下数据类型： ``mindspore.common.dtype.float32`` 或 ``mindspore.common.dtype.float16``。
        - **fno_compute_dtype** (dtype.Number) - FNO skip处的计算类型。默认值： ``mindspore.common.dtype.float32``。支持以下数据类型： ``mindspore.common.dtype.float32`` 或 ``mindspore.common.dtype.float16``。GPU后端建议使用float32，Ascend后端建议使用float16。

    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, resolutions[0], resolutions[1], resolutions[3], in\_channels)` 的Tensor。

    输出：
        Tensor，FNO网络的输出。

        - **output** (Tensor) - shape为 :math:`(batch\_size, resolutions[0], resolutions[1], resolutions[3], out\_channels)` 的Tensor。

    异常：
        - **TypeError** - 如果 `in_channels` 不是int。
        - **TypeError** - 如果 `out_channels` 不是int。
        - **TypeError** - 如果 `hidden_channels` 不是int。
        - **TypeError** - 如果 `lifting_channels` 不是int。
        - **TypeError** - 如果 `projection_channels` 不是int。
        - **TypeError** - 如果 `n_layers` 不是int。
        - **TypeError** - 如果 `data_format` 不是str。
        - **TypeError** - 如果 `add_residual` 不是bool。
        - **TypeError** - 如果 `positional_embedding` 不是bool。
