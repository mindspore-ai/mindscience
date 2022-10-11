mindsponge.system.Protein
=========================

.. py:class:: mindsponge.system.Protein(pdb, sequence, coordinate, pbc_box, template='protein0.yaml', ignore_hydrogen=True, length_unit)

    蛋白质分子。

    参数：
        - **pdb** (str) - 系统中的原子。
        - **sequence** (list) - 原子种类。
        - **coordinate** (Tensor) - 原子的位置坐标。
        - **pbc_box** (Tensor) - 周期性边界条件的box。
        - **template** (Union[dict, str]) - 残基的模板。
        - **ignore_hydrogen** (bool, 可选) - 是否无视氢原子。
        - **length_unit** (str) - 位置坐标的长度单位。

    符号：
        - **B** - Batch size。
        - **A** - 原子总数。
        - **D** - 模拟系统的维度。