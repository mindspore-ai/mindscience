mindflow.cell.MultiHeadAttention
=================================

.. py:class:: mindflow.cell.MultiHeadAttention(in_channels, num_heads, enable_flash_attn=False, fa_dtype=mstype.bfloat16, drop_mode='dropout', dropout_rate=0.0, compute_dtype=mstype.float32)

    多头注意力机制，具体细节可以参见 `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ 。

    参数：
        - **in_channels** (int) - 输入的输入特征维度。
        - **num_heads** (int) - 输出的输出特征维度。
        - **enable_flash_attn** (bool) - 是否使能FlashAttention。FlashAttention只支持 `Ascend` 后端。具体细节参见 `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`_ 。
          默认值： ``False`` 。
        - **fa_dtype** (mindspore.dtype) - FlashAttention计算类型。支持以下类型： `mstype.bfloat16`、 `mstype.float16`。默认值： ``mstype.bfloat16`` ，表示 ``mindspore.bfloat16`` 。
        - **drop_mode** (str) - dropout方式。默认值： ``dropout`` 。支持以下类型： ``dropout`` 和 ``droppath`` 。
        - **dropout_rate** (float) - dropout层丢弃的比率。取值在 `[0, 1]` 。默认值： ``0.0`` 。
        - **compute_dtype** (mindspore.dtype) - 网络层的数据类型。默认值： ``mstype.float32`` ，表示 ``mindspore.float32`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(batch\_size, sequence\_len, in\_channels)` 的Tensor。
        - **attn_mask** (Tensor，可选) - shape为 :math:`(sequence\_len, sequence\_len)` 或
          :math:`(batch\_size, 1, sequence\_len, sequence\_len)` 的Tensor。默认值： ``None`` 。
        - **key_padding_mask** (Tensor，可选) - shape为 :math:`(batch\_size, sequence\_len)` 的Tensor。默认值： ``None`` 。

    输出：
        - **output** (Tensor) - shape为 :math:`(batch\_size, sequence\_len, in\_channels)` 的Tensor。
