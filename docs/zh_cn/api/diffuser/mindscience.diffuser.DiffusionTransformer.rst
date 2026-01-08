mindscience.diffuser.DiffusionTransformer
===========================================

.. py:class:: mindscience.diffuser.DiffusionTransformer(in_channels, out_channels, hidden_channels, layers, heads, time_token_cond=True, compute_dtype=mstype.float32)

    基于 Transformer 主干结构的扩散模型实现。

    参数：
        - **in_channels** (int) - 输入特征维度。
        - **out_channels** (int) - 输出特征维度。
        - **hidden_channels** (int) - 隐藏层特征维度。
        - **layers** (int) - `Transformer` 模块的层数。
        - **heads** (int) - `Transformer` 模块的注意力头数。
        - **time_token_cond** (bool, 可选) - 是否将时间作为条件token。默认 ``True``。
        - **compute_dtype** (mindspore.dtype, 可选) - 计算使用的数据类型。支持 ``mstype.float32`` 或 ``mstype.float16``。默认 ``mstype.float32``，表示 ``mindspore.float32``。

    输入：
        - **x** (Tensor) - 网络输入张量。形状为 :math:`(batch\_size, sequence\_len, in\_channels)`。
        - **timestep** (Tensor) - 时间步输入张量。形状为 :math:`(batch\_size,)`。

    输出：
        - **output** (Tensor) - 输出张量。形状为 :math:`(batch\_size, sequence\_len, out\_channels)`。
