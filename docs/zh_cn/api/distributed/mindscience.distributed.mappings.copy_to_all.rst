mindscience.distributed.mappings.copy_to_all
=============================================

.. py:function:: mindscience.distributed.mappings.copy_to_all(x, group)

    将输入转发到指定通信组中的所有卡。

    参数：
        - **x** (Tensor) - 要复制到所有卡的输入张量。
        - **group** (Union[CommGroup, CommGroupBase]) - 操作的通信组。

    返回：
        输入张量（复制到所有卡）。
