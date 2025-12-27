mindscience.distributed.mappings.reduce_scatter_to_sequence
============================================================

.. py:function:: mindscience.distributed.mappings.reduce_scatter_to_sequence(x, group)

    沿第一个维度对切分的张量执行 reduce-scatter 操作。

    参数：
        - **x** (Tensor) - 要在序列维度上归约和分散的输入张量。
        - **group** (Union[CommGroup, CommGroupBase]) - 操作的通信组。

    返回：
        当前卡的归约和分散张量。
