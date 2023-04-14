mindsponge.cell.TriangleAttention
=================================

.. py:class:: mindsponge.cell.TriangleAttention(orientation, num_head, key_dim, gating, layer_norm_dim, batch_size=None, slice_num=0)

    三角注意力机制。详细实现过程参考 `TriangleAttention <https://www.nature.com/articles/s41586-021-03819-2>`_ 。

    氨基酸对ij之间的信息通过ij,ik,jk三条边的信息整合，具体分为投影、自注意力和输出三个步骤，首先进行氨基酸对i,j,k输入的投影，获取i,j,k两两之间的q,k,v，然后通过经典多头自注意机制，在ij氨基酸对之间的信息中添加上i，j，k三角形边之间的关系，最后输出。

    参数：
        - **orientation** (int) - 决定三角注意力的方向, 分别作为三角形“starting"和"ending"边的自注意力机制。
        - **num_head** (int) - Attention头的数量。
        - **key_dim** (int) - Attention隐藏层的维度。
        - **gating** (bool) - 判断attention是否经过gating的指示器。
        - **layer_norm_dim** (int) - 归一层的维度。
        - **batch_size** (int) - 三角注意力机制中的batch size参数。默认值： ``None``。
        - **slice_num** (int) - 为了减少内存需要进行切分的数量。默认值： ``0``。

    输入：
        - **pair_act** (Tensor) - pair_act。氨基酸对之间的信息，shape为 :math:`(N_{res}, N_{res}, layer\_norm\_dim)` 。
        - **pair_mask** (Tensor) - 三角注意力层矩阵的mask。shape为 :math:`(N_{res}, N_{res})` 。
        - **index** (Tensor) - 在循环中的索引，只会在有控制流的时候使用。
        - **mask** (Tensor) - 当做layernorm操作的时候pair_act的掩码，shape为 :math:`(N_{res}, N_{res})`，默认值： ``None``。

    输出：
        Tensor。三角注意力层中的pair_act。shape为 :math:`(N_{res}, N_{res}, layer\_norm\_dim)` 。