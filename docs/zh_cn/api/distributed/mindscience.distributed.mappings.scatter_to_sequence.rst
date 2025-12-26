mindscience.distributed.mappings.scatter_to_sequence
=====================================================

.. py:function:: scatter_to_sequence(x, group)

    沿第一个维度将张量分散到不同卡上。

    参数：
        - **x** (Tensor) - 要分散到不同卡上的输入张量。
        - **group** (Union[CommGroup, CommGroupBase]) - 操作的通信组。

    返回：
        沿第一个维度切分对应于当前卡的张量切片。
