mindsponge.cell.OuterProductMean
================================

.. py:class:: mindsponge.cell.OuterProductMean(num_outer_channel, act_dim, num_output_channel, batch_size, slice_num=0)

    计算外积平均。

    参数：
        - **num_outer_channel** (float) - outer的通道数量。
        - **act_dim** (int) - act的最后一维的长度。
        - **num_output_channel** (int) - 输出的通道数量。
        - **batch_size** (int) - 在outer product mean中的参数的batch size。
        - **slice_num** (int) - 当内存超出上限时使用的切分数量。默认值：0。

    .. py:method:: compute(left_act, right_act, linear_output_weight, linear_output_bias, d, e)

        计算 pair activation。

        参数：
            - **left_act** (Tensor) - pair activation的左半。
            - **right_act** (Tensor) - pair activation的右半。
            - **linear_output_weight** (Tensor) - 输出权重的参数。
            - **linear_output_bias** (Tensor) - 输出偏置的参数。
            - **d** (int) - 右半pair activationshape的第二根轴。
            - **e** (int) - 右半pair activationshape的第三根轴。

        返回：
            Tensor。Pair activations。