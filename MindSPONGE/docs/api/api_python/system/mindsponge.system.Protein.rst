mindsponge.system.Protein
=========================

.. py:class:: mindsponge.system.Protein(pdb, sequence, coordinate, pbc_box, template='protein0.yaml', ignore_hydrogen=True, length_unit)

    蛋白质分子。

    参数：
        - **pdb** (str) - 系统中的原子。
        - **sequence** (list) - 原子种类。
        - **coordinate** (Tensor) - 原子的位置坐标，shape为(B, A, D)或者(1, A, D)。
        - **pbc_box** (Tensor) - 周期性边界条件的box，shape为(B, D)或者(1, D)。
        - **template** (Union[dict, str]) - 残基的模板。默认文件：'protein0.yaml'。
        - **ignore_hydrogen** (bool, 可选) - 是否无视氢原子。默认值：""True。
        - **length_unit** (str) - 位置坐标的长度单位。

    符号：
        - **B** - Batch size。
        - **A** - 原子总数。
        - **D** - 模拟系统的维度。