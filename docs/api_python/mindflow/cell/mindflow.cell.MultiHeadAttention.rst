mindflow.cell.MultiHeadAttention
=================================

.. py:class:: mindflow.cell.MultiHeadAttention(in_channels, num_heads, drop_mode='dropout', dropout_rate=0.0, compute_dtype=mstype.float32)

    多头注意力机制，具体细节可以参见 `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ 。

    参数：
        - **in_channels** (int) - 输入的输入特征维度。
        - **num_heads** (int) - 输出的输出特征维度。
        - **drop_mode** (str) - dropout方式。默认值： ``dropout`` 。支持以下类型： ``dropout`` 和 ``droppath`` 。
        - **dropout_rate** (float) - dropout层丢弃的比率。取值在 `[0, 1]` 。默认值： ``0.0`` 。
        - **compute_dtype** (mindspore.dtype) - 网络层的数据类型。默认值： ``mstype.float32`` ，表示 ``mindspore.float32`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, sequence\_len, in\_channels)` 的Tensor。
        - **attn_mask** (Tensor) - shape为 :math:`(batch\_size, sequence\_len, sequence\_len)` 或
          :math:`(sequence\_len, sequence\_len)` or :math:`(batch\_size, num_heads, sequence\_len, sequence\_len)` 的Tensor.
        - **key_padding_mask** (Tensor) - shape为 :math:`(batch\_size, sequence\_len)` 或
          :math:`(batch\_size, sequence\_len, sequence\_len)` 或 :math:`(batch\_size, num_heads, sequence\_len, sequence\_len)` 的Tensor.

    输出：
        - **output** (Tensor) - shape为 :math:`(batch\_size, sequence\_len, in\_channels)` 的Tensor。
