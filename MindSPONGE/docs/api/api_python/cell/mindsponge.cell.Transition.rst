mindsponge.cell.Transition
==========================

.. py:class:: mindsponge.cell.Transition(num_intermediate_factor, layer_norm_dim, batch_size, slice_num=0)

    转换层。

    参数：
        - **num_intermediate_factor** (float) - 中间变量的数量。
        - **layer_norm_dim** (int) - 归一层的最后一维长度。
        - **batch_size** (int) - 转换层的batch size。
        - **slice_num** (int) - 当内存超出上限时在转换层使用的切分数量。默认值：0。

    .. py:method:: compute(act, transition1_weight, transition1_bias, transition2_weight, transition2_bias)

        计算pair activation。

        参数：
            - **act** (Tensor) - Pair activations。
            - **transition1_weight** (float) - transition1的权重。
            - **transition1_bias** (float) - transition1的偏置。
            - **transition2_weight** (float) - transition2的权重。
            - **transition2_bias** (float) - transition2的偏置。

        返回：
            Tensor。Pair activations。