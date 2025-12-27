mindscience.distributed.mappings.all_to_all_from_sequence_to_hidden
====================================================================

.. py:function:: mindscience.distributed.mappings.all_to_all_from_sequence_to_hidden(x, group)

    执行从序列维度切分到特征维度切分的 all-to-all 操作。

    参数：
        - **x** (Tensor) - 序列维度切分的输入张量。
        - **group** (Union[CommGroup, CommGroupBase]) - 操作的通信组。

    返回：
        从序列维度切分到特征维度切分的张量。
