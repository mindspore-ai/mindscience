mindscience.distributed.mappings.gather_from_hidden
====================================================

.. py:function:: mindscience.distributed.mappings.gather_from_hidden(x, group)

    沿最后一个维度收集切分的张量。

    参数：
        - **x** (Tensor) - 沿最后一个维度切分的输入张量。
        - **group** (Union[CommGroup, CommGroupBase]) - 操作的通信组。

    返回：
        沿最后一个维度收集所有切分的张量。
