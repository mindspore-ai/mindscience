mindsponge.cell.Transition
==========================

.. py:class:: mindsponge.cell.Transition(num_intermediate_factor, input_dim, batch_size=None, slice_num=0)

    两层全连接层，中间输出特征数为输入特征数的倍数。

    .. math::
        Transition(\mathbf{act}) = Linear(Linear(\mathbf{act}))

    参数：
        - **num_intermediate_factor** (float) - 中间输出的特征数相对于输入特征数的膨胀倍数。
        - **input_dim** (int) - 输入的特征数。
        - **batch_size** (int) - 转换层权重的batch size，应用控制流时需设置该变量，默认值： ``None``。
        - **slice_num** (int) - 当内存超出上限时在转换层使用的切分数量。默认值： ``0``。

    输入：
        - **act** (Tensor) - shape为 :math:`(..., input\_dim)` 的Tensor。
        - **index** (Tensor) - while循环中权重的索引，应用控制流时需设置该变量，默认值： ``None``。
        - **mask** (Tensor) - 当做layernorm操作的时候act的掩码，shape为 :math:`(32, input\_dim)`，默认值： ``None``。

    输出：
        Tensor。shape为 :math:`(..., input\_dim)` 的Tensor。
