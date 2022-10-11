mindsponge.cell.TriangleAttention
=================================

.. py:class:: mindsponge.cell.TriangleAttention(orientation, num_head, key_dim, gating, layer_norm_dim, batch_size, slice_num=0)

    三角注意力机制。

    参数：
        - **orientation** (int) - 决定三角的维度。
        - **num_head** (int) - 头的数量。
        - **key_dim** (int) - 输入的维度。
        - **gating** (bool) - 判断attention是否经过gating的指示器。
        - **layer_norm_dim** (int) - 归一层的维度。
        - **batch_size** (int) - 三角注意力机制中的batch size参数。
        - **slice_num** (int) - 为了减少内存所制作的切分的数量。默认值：0。

    输入：
        - **pair_act** (Tensor) - pair_act。
        - **pair_mask** (Tensor) - 三角注意力层矩阵的mask。shape为(batch_size, num_heads, query_seq_length, value_seq_length)。
        - **index** (Tensor) - 在循环中的索引，只会在有控制流的时候使用。

    输出：
        Tensor。三角注意力层中的pair_act。shape为(batch_size, query_seq_length, hidden_size)。

    .. py:method:: compute(pair_act, input_mask, index, nonbatched_bias)

        将pair_act经过attention层，计算pair_act。

        参数：
            - **pair_act** (Tensor) - pair_act。
            - **input_mask** (Tensor) - 三角注意力层矩阵的mask。shape为(batch_size, num_heads, query_seq_length, value_seq_length)。
            - **index** (Tensor) - 在循环中的索引，只会在有控制流的时候使用。
            - **nonbatched_bias** (Tensor) - 没有batch size维的偏置参数。

        返回：
            Tensor。三角注意力层中的pair_act。shape为(batch_size, query_seq_length, hidden_size)。