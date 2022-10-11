mindsponge.cell.GlobalAttention
===============================

.. py:class:: mindsponge.cell.GlobalAttention(num_head, gating, hidden_size, output_dim, batch_size)

    global gated自注意力机制，具体实现请参考 `Highly accurate protein structure prediction with AlphaFold <https://www.nature.com/articles/s41586-021-03819-2>`_ 。

    参数：
        - **num_head** (int) - 头的数量。
        - **gating** (bool) - 判断attention是否经过gating的指示器。
        - **hidden_size** (int) - 输入的隐藏尺寸。
        - **output_dim** (int) - 输出的最后一维度的长度。
        - **batch_size** (int) - attention中参数的batch size。

    输入：
        - **q_data** (Tensor) - shape为(batch_size, query_seq_length, q_data_dim)的Q Tensor。
        - **m_data** (Tensor) - shape为(batch_size, value_seq_length, m_data_dim)的K和V。
        - **q_mask** (Tensor) - q_data的二元mask，在长度元素中padded的位置为0，其他的位置为1。
        - **attention_mask** (Tensor) - 注意力矩阵的mask。shape为(batch_size, query_seq_length, value_seq_length)。
        - **bias** (Tensor) - attention矩阵的偏置。
        - **index** (Tensor) - 在循环中的索引。

    输出：
        Tensor。Attention层的输出，shape是(batch_size, query_seq_length, hidden_size)。