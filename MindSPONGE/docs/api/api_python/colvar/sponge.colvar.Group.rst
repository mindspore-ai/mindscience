sponge.colvar.Group
=========================

.. py:class:: Group(atoms: Union[List[AtomsBase], Tuple[AtomsBase]], batched: bool = False, keep_in_box: bool = False, axis: int = 1, name: str = 'atoms_group')

    原子组

    参数：
        - **atoms** (Union[List[AtomsBase], Tuple[AtomsBase]]) - AtomsBase列表。成员应该是 AtomsBase 的子类。
        - **batched** (bool) - 索引的第一个维度是否为批大小。默认值：``False``。
        - **keep_in_box** (bool) - 是否在PBC box中替换坐标。默认值：``False``。
        - **axis** (int) - 计算位置平均值的axis。
        - **name** (str) - Colvar的名称。默认值：'atoms_group'
    
    支持的平台：
        ``Ascend`` ``GPU``

    .. py:method:: construct(coordinate: Tensor, pbc_box: Tensor = None)

        获取原子组的位置坐标

        参数：
            - **coordinate** (Tensor) - 张量的shape (B, A, D) 。数据类型为float。原子在系统中的位置坐标。其中，B表示批量大小，即模拟中的步行者数量。A表示系统中的原子数。D表示仿真系统的维度。通常为3。
            - **pbc_box** (Tensor) - 张量的shape (B, D)。数据类型为float。PBC box的张量。默认值：``None``。

        返回：
            位置(Tensor)：张量的shape (B, a_1, a_2, ..., a_{n}, D) 。数据类型为float。a_{i}表示特定原子的维度。