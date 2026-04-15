sponge.colvar.Group
=========================

.. py:class:: sponge.colvar.Group(atoms: Union[List[AtomsBase], Tuple[AtomsBase]], batched: bool = False, keep_in_box: bool = False, axis: int = 1, name: str = 'atoms_group')

    原子组。

    参数：
        - **atoms** (Union[List[AtomsBase], Tuple[AtomsBase]]) - AtomsBase列表。成员应该是 AtomsBase 的子类。
        - **batched** (bool) - 索引的第一个维度是否为批大小。默认值： ``False``。
        - **keep_in_box** (bool) - 是否在PBC box中替换坐标。默认值： ``False``。
        - **axis** (int) - 计算位置平均值的axis。
        - **name** (str) - Colvar的名称。默认值：'atoms_group'。