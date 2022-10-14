mindsponge.cell.InvariantPointAttention
=======================================

.. py:class:: mindsponge.cell.InvariantPointAttention(num_head, num_scalar_qk, num_scalar_v, num_point_v, num_point_qk, num_channel, pair_dim)

    不变点注意力模块。

    参数：
        - **num_head** (int) - 头的数量。
        - **num_scalar_qk** (int) - scalar query/key的数量。
        - **num_scalar_v** (int) - scalar value的数量。
        - **num_point_v** (int) - point value的数量。
        - **num_point_qk** (int) - point query/key的数量。
        - **num_channel** (int) - 通道数量。
        - **pair_dim** (int) - pair的最后一维长度。