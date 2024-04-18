sponge.colvar.AllAtoms
============================

.. py:class:: sponge.colvar.AllAtoms(system=None, num_atoms: int = None, keep_in_box: bool = False, dimension: int = 3, name: str = 'all_atoms')

    模拟系统的所有原子。

    参数：
        - **system** (Molecule) - 模拟系统。默认值： ``None``。
        - **num_atoms** (int) - 原子的数量。当 `system` 为空时，必须给出原子的数量。默认值： ``None``。
        - **keep_in_box** (bool) - 是否在PBC框中替换坐标。默认值： ``False``。
        - **dimension** (int) - 仿真系统的空间维度。默认值：3。
        - **name** (str) - Colvar 的名称。默认值：'all_atoms'。