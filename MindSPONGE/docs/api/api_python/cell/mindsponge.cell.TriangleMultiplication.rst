mindsponge.cell.TriangleMultiplication
======================================

.. py:class:: mindsponge.cell.TriangleMultiplication(num_intermediate_channel, equation, layer_norm_dim, batch_size=None)

    三角乘法层。详细实现过程参考 `TriangleMultiplication <https://www.nature.com/articles/s41586-021-03819-2>`_ 。
    氨基酸对ij之间的信息通过ij,ik,jk三条边的信息整合，将ik和jk的点乘结果信息添加到ij边。

    参数：
        - **num_intermediate_channel** (float) - 中间通道的数量。
        - **equation** (str) - 三角形边顺序的爱因斯坦算符表示，分别对应于"incoming"和"outgoing"的边更新形式。 :math:`(ikc,jkc->ijc, kjc,kic->ijc)`。
        - **layer_norm_dim** (int) - 归一层的最后一维的长度。
        - **batch_size** (int) - 三角乘法中的batch size。默认值："None"。

    输入：
        - **pair_act** (Tensor) - pair_act。氨基酸对之间的信息，shape为 :math:`(N_{res}, N_{res}, layer\_norm\_dim)` 。
        - **pair_mask** (Tensor) - 三角乘法层矩阵的mask。shape为 :math:`(N_{res}, N_{res})` 。
        - **index** (Tensor) - 在循环中的索引，只会在有控制流的时候使用。

    输出：
        Tensor。三角乘法层中的pair_act。shape为 :math:`(N_{res}, N_{res}, layer\_norm\_dim)` 。