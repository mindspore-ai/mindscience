mindsponge.cell.MSAColumnGlobalAttention
========================================

.. py:class:: mindsponge.cell.MSAColumnGlobalAttention(num_head, gating, msa_act_dim, batch_size, slice_num=0)

    MSA列全局注意力层。

    参数：
        - **num_head** (int) - 头的数量。
        - **gating** (bool) - 判断attention是否经过gating的指示器。
        - **msa_act_dim** (int) - `msa_act` 的维度。msa_act为AlphaFold模型中MSA检索后所使用的中间变量。
        - **batch_size** (int) - MSAColumnAttention中参数的batch size。
        - **slice_num** (int) - 为了减少内存需要进行切分的数量。默认值0。

    输入：
        - **msa_act** (Tensor) - msa_act，AlphaFold模型中MSA检索后所使用的中间变量。
        - **msa_mask** (Tensor) - MSAColumnAttention矩阵的mask，shape为(batch_size, num_heads, query_seq_length, value_seq_length)。
        - **index** (Tensor) - 在循环中的索引，只会在有控制流的时候使用。

    输出：
        Tensor。MSAColumnAttention层的输出msa_act，shape是(batch_size, query_seq_length, hidden_size)。

    .. py:method:: compute(msa_act, input_mask, index)

        将 `msa_act` 经过attention层，进行计算。

        参数：
            - **msa_act** (Tensor) - msa_act，AlphaFold模型中MSA检索后所使用的中间变量。
            - **input_mask** (Tensor) - MSAColumnAttention矩阵的mask，shape为(batch_size, num_heads, query_seq_length, value_seq_length)。
            - **index** (Tensor) - 在循环中的索引，只会在有控制流的时候使用。

        返回：
            Tensor。Attention层的输出msa_act，shape是(batch_size, query_seq_length, hidden_size)。
