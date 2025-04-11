mindflow.cell.SNO
=========================

.. py:class:: mindflow.cell.SNO(in_channels, out_channels, hidden_channels=64, num_sno_layers=3, data_format="channels_first", transforms=None, kernel_size=5, num_usno_layers=0, num_unet_strides=1, activation="gelu", compute_dtype=mstype.float32)

    谱神经算子（Spectral Neural Operator, SNO）基类，包含一个提升层（编码器）、多个谱变换层（谱空间的线性变换）和一个投影层（解码器）。
    这是一种类似FNO的架构，但使用多项式变换（Chebyshev、Legendre等）替代傅里叶变换。
    详细信息请参考谱神经算子论文 `Spectral Neural Operators <https://arxiv.org/pdf/2205.10573>`_ 。

    参数：
        - **in_channels** (int) - 输入中的通道数。
        - **out_channels** (int) - 输出中的通道数。
        - **hidden_channels** (int) - SNO层输入和输出的通道数。默认值： ``64``。
        - **num_sno_layers** (int) - 谱层数量。默认值： ``3``。
        - **data_format** (str) - 输入数据的通道顺序。默认值： ``channels_first``。
        - **transforms** (list(list(mindspore.Tensor))) - 沿x、y、z轴的正变换和逆多项式变换列表。结构形式为：[[transform_x, inv_transform_x], [transform_z, inv_transform_z]]。变换矩阵形状应为(n_modes, resolution)，其中n_modes为多项式变换模式数，resolution为对应方向输入的空间分辨率。逆变换矩阵形状为(resolution, n_modes)。默认值： ``None``。
        - **kernel_size** (int) - 指定SNO层中卷积核的高度和宽度。默认值： ``5``。
        - **num_usno_layers** (int) - 带UNet跳跃连接的谱层数量。默认值： ``0``。
        - **num_unet_strides** (int) - UNet跳跃连接中卷积下采样块的数量。默认值： ``1``。
        - **activation** (Union[str, class]) - 激活函数，支持字符串或类形式。默认值： ``gelu``。
        - **compute_dtype** (dtype.Number) - 计算数据类型。默认值： ``mstype.float32``。
          可选 ``mstype.float32`` 或 ``mstype.float16``。GPU后端推荐使用float32，Ascend后端推荐使用float16。

    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, in_channels, resolution)` 的张量。

    输出：
        - shape为 :math:`(batch\_size, out_channels, resolution)` 的张量。

    异常：
        - **TypeError** - 如果 `in_channels` 不是int。
        - **TypeError** - 如果 `out_channels` 不是int。
        - **TypeError** - 如果 `hidden_channels` 不是int。
        - **TypeError** - 如果 `num_sno_layers` 不是int。
        - **TypeError** - 如果 `transforms` 不是list。
        - **ValueError** - 如果 `transforms` 长度不在(1, 2, 3)范围内。
        - **TypeError** - 如果 `num_usno_layers` 不是int。
