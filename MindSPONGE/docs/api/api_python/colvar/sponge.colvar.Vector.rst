sponge.colvar.Vector
========================

.. py:class:: Vector(atoms: AtomsBase = None, atoms0: AtomsBase = None, atoms1: AtomsBase = None, batched: bool = False, use_pbc: bool = None, keepdims: bool = None, axis: int = -2, name: str = 'vector')

    特定原子或虚拟原子之间的向量

    参数：
        - **atoms** (AtomsBase) - shape为 (..., 2, D) 的原子形成shape为 (..., D) 或 (..., 1, D) 的向量。不能与`atoms0`或`atoms1`一起使用。默认值:``None``。其中，D表示仿真系统的维度。通常为3。
        - **atoms0** (AtomsBase) - shape为 (..., D) 的原子的初始点形成shape (..., D) 的向量。 必须与`atoms1`一起使用，不能与`atoms`一起使用。默认值：``None``。
        - **atoms1** (AtomsBase) - shape为 (..., D) 的原子的端点，形成shape为 (..., D) 的向量。必须与`atoms0`一起使用，不能与`atoms`一起使用。默认值：``None``。
        - **batched** (bool) - 索引的第一个维度是否为批大小。默认值：``False``。
        - **use_pbc** (bool) - 是否在周期边界条件下计算距离。默认值：``None``。
        - **keepdims** (bool) - 如果设置为 True，则取自`atoms`的轴将保留，向量的shape将是 (..., 1, D) 。如果设置为 False，则向量的shape将为 (..., D) 。如果为 None，则其值将根据输入原子：如果秩大于 2，则为False，否则为 True。仅在使用`atoms`初始化时有效。默认值：``None``。
        - **axis** (int) - 沿其取原子坐标的轴，其维度必须为 2。它仅在使用`atoms`初始化时有效。默认值：-2。
        - **name** (str) - Colvar的名称。默认值：'vector'

    支持的平台：
        ``Ascend`` ``GPU``
    
    .. py:method:: ndim()

        向量的秩（维数）

    .. py:method:: shape()

        向量的shape

    .. py:method:: construct(coordinate: Tensor, pbc_box: Tensor = None)

        获取特定原子或虚拟原子之间的向量

        参数：
            - **coordinate** (Tensor) - 张量的shape (B, A, D) 。数据类型为float。其中，B表示批量大小，即模拟中的步行者数量。A表示系统中的原子数。
            - **pbc_box** (Tensor) - 张量的shape (B, D) 。数据类型为float。默认值：``None``。

        返回：
            向量(Tensor): 张量的shape (B, ..., D) 。数据类型为float。