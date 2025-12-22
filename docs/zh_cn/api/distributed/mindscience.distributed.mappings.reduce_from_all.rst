mindscience.distributed.mappings.reduce_from_all
=================================================

.. py:function:: reduce_from_all(x, group)

    对所有卡执行全归约操作。

    参数：
        - **x** (Tensor) - 要在所有卡中归约的输入张量。
        - **group** (Union[CommGroup, CommGroupBase]) - 操作的通信组。

    返回：
        来自所有卡的归约张量。
