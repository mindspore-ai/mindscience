mindsponge.system.Protein
=========================

.. py:class:: mindsponge.system.Protein(pdb=None, sequence=None, coordinate=None, pbc_box=None, template='protein0.yaml', ignore_hydrogen=True, length_unit=None)

    蛋白质分子。

    参数：
        - **pdb** (str) - 系统中的原子，list中的数据类型可以是str或int。默认值："None"。
        - **sequence** (list) - 原子种类，数据类型为str，可以是ndarry或者是list。默认值："None"。
        - **coordinate** (Tensor) - 原子的位置坐标，shape为(B, A, D)或者(1, A, D)，数据类型为float。默认值："None"。
        - **pbc_box** (Tensor) - 周期性边界条件的box，shape为(B, D)或者(1, D)，数据类型为float。默认值："None"。
        - **template** (Union[dict, str]) - 残基的模板。字典的key为"base", "template", 分子的名字等。value为文件的名字。默认文件：'protein0.yaml'。
        - **ignore_hydrogen** (bool, 可选) - 是否无视氢原子。默认值："True"。
        - **length_unit** (str) - 位置坐标的长度单位。默认值："None"。

    符号：
        - **B** - Batch size。
        - **A** - 原子总数。
        - **D** - 模拟系统的维度，一般为3。