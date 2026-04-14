sponge.colvar.get_atoms
===========================

.. py:function:: sponge.colvar.get_atoms(atoms: Union[AtomsBase, List[AtomsBase], Tuple[AtomsBase], Tensor, Parameter, ndarray], batched: bool = False, keep_in_box: bool = False)

    获取原子或组。

    参数：
        - **atoms** (Union[list, tuple, AtomsBase, Tensor, Parameter, ndarray]) - 原子的列表。
        - **batched** (bool) - 索引的第一个维度是否为批量大小。默认值： ``False``。
        - **keep_in_box** (bool) - 是否置换PBC box中的坐标。默认值： ``False``。

    返回：
        atoms(Union[Atoms, Group])：原子或组。