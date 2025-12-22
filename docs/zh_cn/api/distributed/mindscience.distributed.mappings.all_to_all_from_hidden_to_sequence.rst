mindscience.distributed.mappings.all_to_all_from_hidden_to_sequence
====================================================================

.. py:function:: all_to_all_from_hidden_to_sequence(x, group)

    执行从特征维度切分到序列维度切分的 all-to-all 操作。

    参数：
        - **x** (Tensor) - 特征维度切分的输入张量。
        - **group** (Union[CommGroup, CommGroupBase]) - 操作的通信组。

    返回：
        从特征维度切分到序列维度切分的张量。
