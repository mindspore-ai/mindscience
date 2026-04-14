mindsponge.cell.MSARowAttentionWithPairBias
===========================================

.. py:class:: mindsponge.cell.MSARowAttentionWithPairBias(num_head, key_dim, gating, msa_act_dim, pair_act_dim, batch_size=None, slice_num=0)

    MSA行注意力层。具体实现参考 `Jumper et al. (2021) Suppl. Alg. 7 'MSARowAttentionWithPairBias' <https://www.nature.com/articles/s41586-021-03819-2>`_ 。来自pair激活值的信息作为MSARowAttention的注意力矩阵的偏置项，这样可以利用pair信息更新msa表示的状态。

    参数：
        - **num_head** (int) - attention头的数量。
        - **key_dim** (int) - attention隐藏层的维度。
        - **gating** (bool) - 判断attention是否经过gating的指示器。
        - **msa_act_dim** (int) - msa_act的维度。
        - **pair_act_dim** (int) - pair_act的维度。
        - **batch_size** (int) - MSARowAttentionWithPairBias中参数的batch size，控制流场景下使用。默认值： ``None``。
        - **slice_num** (int) - 为了减少内存需要进行切分的数量。默认值： ``0``。

    输入：
        - **msa_act** (Tensor) - shape为 :math:`(N_{seqs}, N_{res}, msa\_act\_dim)` 。
        - **msa_mask** (Tensor) - msa_act矩阵的掩码，shape为 :math:`(N_{seqs}, N_{res})` 。
        - **pair_act** (Tensor) - shape为 :math:`(N_{res}, N_{res}, pair\_act\_dim)` 。
        - **index** (Tensor) - 在循环中的索引，只会在有控制流的时候使用。默认值： ``None``。 
        - **norm_msa_mask** (Tensor) - 当做layernorm操作的时候msa_act的掩码，shape为 :math:`(N_{seqs}, N_{res})`，默认值： ``None``。
        - **norm_pair_mask** (Tensor) - 当做layernorm操作的时候pair_act的掩码，shape为 :math:`(N_{res}, N_{res})`，默认值： ``None``。
        - **res_idx** (Tensor) - 用于执行ROPE的残基索引，shape为 :math:`(N_{res}, )`，默认值： ``None``。

    输出：
        Tensor。本层输出的msa_act，shape是 :math:`(N_{seqs}, N_{res}, msa\_act\_dim)` 。
