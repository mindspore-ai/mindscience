mindsponge.data.get_template_index
==================================

.. py:class:: mindsponge.data.get_template_index(template, names, key="atom_name")

    根据原子名称获取系统中的原子序号。

    参数：
        - **template** (dict) - 模板的文件名称。
        - **names** (ndarray) - 残基名称。
        - **key** (str) - 原子名称。默认值："atom_name"。

    输出：
        ndarray。系统中原子索引。