mindsponge.cell.MSAColumnGlobalAttention
========================================

.. py:class:: mindsponge.cell.MSAColumnGlobalAttention(num_head, gating, msa_act_dim, batch_size=None, slice_num=0)

    MSA列全局注意力层。详细实现过程参考 `Jumper et al. (2021) Suppl. Alg. 19 'MSAColumnGlobalAttention' <https://www.nature.com/articles/s41586-021-03819-2>`_ 。
    将输入的msa信息在序列与残基轴上做转置，而后调用 `GlobalAttention <https://www.mindspore.cn/mindsponge/docs/zh-CN/master/cell/mindsponge.cell.GlobalAttention.html>`_ ，在输入的多条序列之间做attention操作，不会处理序列本身残基之间的关系。相比较于MSAColumnAttention，它使用全局的注意力机制，可以处理更大规模的输入序列。

    参数：
        - **num_head** (int) - attention头的数量。
        - **gating** (bool) - 判断attention是否经过gating的指示器。
        - **msa_act_dim** (int) - 输入msa_act的维度。
        - **batch_size** (int) - MSAColumnGlobalAttention中参数的batch size，在控制流场景下使用。默认值： ``None``。
        - **slice_num** (int) - 为了减少内存需要进行切分的数量。默认值： ``0``。

    输入：
        - **msa_act** (Tensor) - shape为 :math:`(N_{seqs}, N_{res}, msa\_act\_dim)` 。
        - **msa_mask** (Tensor) - msa_act矩阵的mask，shape为 :math:`(N_{seqs}, N_{res})` 。
        - **index** (Tensor) - 在循环中的索引，只会在有控制流的时候使用。默认值为： ``None``。

    输出：
        Tensor。本层输出的msa_act，shape是 :math:`(N_{seqs}, N_{res}, msa\_act\_dim)` 。