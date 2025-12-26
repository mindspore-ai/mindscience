mindscience.distributed.mappings.scatter_to_hidden
===================================================

.. py:function:: scatter_to_hidden(x, group)

    沿最后一个维度将张量分散到不同卡上。

    参数：
        - **x** (Tensor) - 要分散到不同卡上的输入张量。
        - **group** (Union[CommGroup, CommGroupBase]) - 操作的通信组。

    返回：
        对应于当前卡的张量切片。
