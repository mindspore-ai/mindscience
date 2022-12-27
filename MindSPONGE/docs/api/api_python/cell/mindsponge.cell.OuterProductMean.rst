mindsponge.cell.OuterProductMean
================================

.. py:class:: mindsponge.cell.OuterProductMean(num_outer_channel, act_dim, num_output_channel, batch_size=None, slice_num=0)

    通过外积平均计算输入张量（act）在第二维上的相关性，得到的相关性可以用于更新相关特征（如Pair特征）。

    .. math::
        OuterProductMean(\mathbf{act}) = Linear(flatten(mean(\mathbf{act}\otimes\mathbf{act})))

    参数：
        - **num_outer_channel** (float) - OuterProductMean中间层的通道数量。
        - **act_dim** (int) - 输入act的最后一维的长度。
        - **num_output_channel** (int) - 输出的通道数量。
        - **batch_size** (int) - OuterProductMean中的参数的batch size，应用while控制流时需要设置该变量， 默认值："None"。
        - **slice_num** (int) - 当内存超出上限时使用的切分数量。默认值：0。

    输入：
        - **act** (Tensor) - 维度为 :math:`(dim_1, dim_2, act\_dim)`。
        - **mask** (Tensor) - OuterProductMean的mask，shape为 :math:`(dim_1, dim_2)`。
        - **mask_norm** (Tensor) - mask沿第一根轴的L2-norm的平方，预先计算避免在循环重复计算。shape为 :math:`(dim_2, dim_2, 1)`。
        - **index** (Tensor) - 在循环中的索引。默认值："None"。

    输出：
        Tensor。OuterProductMean的输出，shape是 :math:`(dim_2, dim_2, num\_output\_channel)`。
