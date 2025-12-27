mindscience.models.transformer.MultiHeadAttention
===================================================

.. py:class:: mindscience.models.transformer.MultiHeadAttention(in_channels, num_heads, enable_flash_attn=False, fa_dtype=mstype.bfloat16, drop_mode="dropout", dropout_rate=0.0, compute_dtype=mstype.float32)

    多头注意力，提出于 `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_。

    参数:
        - **in_channels** (int) - 输入通道。
        - **num_heads** (int) - 注意力头的数量。
        - **enable_flash_attn** (bool) - 是否使用闪存注意力。闪存注意力仅支持 Ascend 后端。闪存注意力提出于 `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`_。默认值： ``False``。
        - **fa_dtype** (mindspore.dtype) - FlashAttention 计算数据类型。从 `mstype.bfloat16`、`mstype.float16` 中选择。默认值： ``mstype.bfloat16``，表示 ``mindspore.bfloat16``。
        - **drop_mode** (str) - 丢弃方法，支持 ``"dropout"`` 或 ``"droppath"``。默认值： ``"dropout"``。
        - **dropout_rate** (float) - dropout 层的丢弃率，大于 0 且小于等于 1。默认值： ``0.0``。
        - **compute_dtype** (mindspore.dtype) - 计算数据类型。默认值： ``mstype.float32``，表示 ``mindspore.float32``。

    输入:
        - **x** (Tensor) - Tensor，形状为 :math:`(batch\_size, sequence\_len, in\_channels)`。
        - **attn_mask** (Tensor, 可选) - Tensor，形状为 :math:`(sequence\_len, sequence\_len)` 或 :math:`(batch\_size, 1, sequence\_len, sequence\_len)`。默认值： ``None``。
        - **key_padding_mask** (Tensor, 可选) - Tensor，形状为 :math:`(batch\_size, sequence\_len)`。默认值： ``None``。

    输出:
        - **output** (Tensor) - 形状为 :math:`(batch\_size, sequence\_len, in\_channels)`。
