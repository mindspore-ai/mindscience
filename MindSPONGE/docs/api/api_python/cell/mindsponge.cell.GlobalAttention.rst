mindsponge.cell.GlobalAttention
===============================

.. py:class:: mindsponge.cell.GlobalAttention(num_head, gating, input_dim, output_dim, batch_size=None)

    global gated自注意力机制，具体实现请参考 `Highly accurate protein structure prediction with AlphaFold <https://www.nature.com/articles/s41586-021-03819-2>`_ 。对于GlobalAttention模块，query/key/value tensor的shape需保持一致。

    参数：
        - **num_head** (int) - 头的数量。
        - **gating** (bool) - 判断attention是否经过gating的指示器。
        - **input_dim** (int) - 输入的最后一维的长度。
        - **output_dim** (int) - 输出的最后一维的长度。
        - **batch_size** (int) - attention中权重的batch size，仅在有while控制流时使用，默认值："None"。

    输入：
        - **q_data** (Tensor) - shape为(batch_size, seq_length, input_dim)的query tensor，其中seq_length是query向量的序列长度。
        - **m_data** (Tensor) - shape为(batch_size, seq_length, input_dim)的key和value tensor。
        - **q_mask** (Tensor) - shape为(batch_size, seq_length, 1)的q_data的mask。
        - **bias** (Tensor) - attention矩阵的偏置。默认值："None"。
        - **index** (Tensor) - 在while循环中的索引，仅在有while控制流时使用。默认值："None"。

    输出：
        Tensor。GlobalAttention层的输出tensor，shape是(batch_size, seq_length, output_dim)。