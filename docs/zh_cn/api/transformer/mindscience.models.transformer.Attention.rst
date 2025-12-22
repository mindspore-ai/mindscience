mindscience.models.transformer.Attention
===================================================

.. py:class:: mindscience.models.transformer.Attention(in_channels, num_heads, compute_dtype=mstype.float32)

    注意力实现基类

    参数：
        - **in_channels** (int) - 输入向量的维度。
        - **num_heads** (int) - 注意力头的数量。
        - **compute_dtype** (mindspore.dtype, 可选) - 计算数据类型。默认值： ``mstype.float32`` ，表示 ``mindspore.float32`` 。

    输入：
        - **x** (Tensor) - 形状为 :math:`(batch\_size, sequence\_len, in\_channels)` 的张量。
        - **attn_mask** (Tensor, 可选) - 形状为 :math:`(sequence\_len, sequence\_len)` 或 :math:`(batch\_size, 1, sequence\_len, sequence\_len)` 的张量。默认值： ``None`` 。
        - **key_padding_mask** (Tensor, 可选) - 形状为 :math:`(batch\_size, sequence\_len)` 的张量。默认值： ``None`` 。

    输出：
        - **output** (Tensor) - 形状为 :math:`(batch\_size, sequence\_len, in\_channels)` 的张量。
