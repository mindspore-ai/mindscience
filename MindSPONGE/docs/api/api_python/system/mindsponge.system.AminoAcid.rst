mindsponge.system.AminoAcid
===========================

.. py:class:: mindsponge.system.AminoAcid(name=' ', template=None, atom_name=None, start_index=0)

    氨基酸的残基。

    参数：
        - **name** (str) - 残基名称。默认值：" "。
        - **template** (dict 或 str) - 残基的模板。默认值："None"。
        - **atom_name** (list) - 原子名称。默认值："None"。
        - **start_index** (int) - 在残基中第一个原子的开始序号。默认值：0。

    输出：
        - **B** - Batch size。
        - **A** - 原子总数。
        - **b** - 边总数。