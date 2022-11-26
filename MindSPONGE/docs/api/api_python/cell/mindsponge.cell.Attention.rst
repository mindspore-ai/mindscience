mindsponge.cell.Attention
=========================

.. py:class:: mindsponge.cell.Attention(num_head, hidden_size, gating, q_data_dim, m_data_dim, output_dim, batch_size=None)

    多头注意力机制，具体实现请参考 `Attention is all you need <https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`_ 。Attention公式如下，query向量长度与输入一致，key向量长度为key长度和目标长度。

    .. math::
        Attention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

    :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)` 。默认有偏置。如果query，key和value张量相同，则表现为self attention。

    参数：
        - **num_head** (int) - 头的数量。
        - **hidden_size** (int) - 输入的隐藏尺寸。
        - **gating** (bool) - 判断attention是否经过gating的指示器。
        - **q_data_dim** (int) - query的最后一维度的长度。
        - **m_data_dim** (int) - key和value最后一维度的长度。
        - **output_dim** (int) - 输出的最后一维度的长度。
        - **batch_size** (int) - attention中权重的batch size，仅在有while控制流时使用，默认值："None"。

    输入：
        - **q_data** (Tensor) - shape为(batch_size, query_seq_length, q_data_dim)的query Tensor，其中query_seq_length是query向量的序列长度。
        - **m_data** (Tensor) - shape为(batch_size, value_seq_length, m_data_dim)的key和value Tensor，其中value_seq_length是value向量的序列长度。
        - **attention_mask** (Tensor) - 注意力矩阵的mask。shape为(batch_size, num_heads, query_seq_length, value_seq_length)。
        - **index** (Tensor) - 在while循环中的索引，仅在有while控制流时使用。默认值："None"。
        - **nonbatched_bias** (Tensor) - attention矩阵中无batch维的偏置。shape为(num_heads, query_seq_length, value_seq_length)。默认值："None"。

    输出：
        Tensor。Attention层的输出tensor，shape是(batch_size, query_seq_length, hidden_size)。