mindsponge.cell.TriangleMultiplication
======================================

.. py:class:: mindsponge.cell.TriangleMultiplication(num_intermediate_channel, equation, layer_norm_dim, batch_size)

    三角乘法层。

    参数：
        - **num_intermediate_channel** (float) - 中间通道的数量。
        - **equation** (str) - 在该层网络中用到的公式。
        - **layer_norm_dim** (int) - 归一层的最后一维的长度。
        - **batch_size** (int) - 三角乘法中的batch size。