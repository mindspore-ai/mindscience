mindsponge.system.Residue
=========================

.. py:class:: mindsponge.system.Residue(atom_name=None, atom_type=None, atom_mass=None, atom_charge=None, atomic_number=None, bond=None, head_atom=None, tail_atom=None, start_index=0, name='MOL', template=None)

    小分子中残基的类。

    参数：
        - **atom_name** (list) - 原子名称。默认值："None"。
        - **atom_type** (list) - 原子种类。默认值："None"。
        - **atom_mass** (Tensor) - 原子质量。默认值："None"。
        - **atom_charge** (Tensor) - 原子电荷。默认值："None"。
        - **atomic_number** (Tensor) - 原子序数。默认值："None"。
        - **bond** (Tensor) - 边序号。默认值："None"。
        - **head_atom** (int) - 与前一个残基相连接的头原子的索引。默认值："None"。
        - **tail_atom** (int) - 与下一个残基相连的尾原子的索引。默认值："None"。
        - **start_index** (int) - 残基中第一个原子的开始索引。默认值：0。
        - **name** (str) - 残基名称。默认值：'MOL'。
        - **template** (Union[dict, str]) - 残基的模板。默认值："None"。

    符号：
        - **B** - Batch size。
        - **A** - 原子总数。
        - **b** - 边总数。

    .. py:method:: add_atom(atom_name=None, atom_type=None, atom_mass=None, atom_charge=None, atomic_number=None)

        设定原子。

        参数：
            - **atom_name** (Union[numpy.ndarray, list(str)]) - 原子名称。默认值："None"。
            - **atom_type** (Union[numpy.ndarray, list(str)]) - 原子种类。默认值："None"。
            - **atom_mass** (Tensor) - 原子质量。默认值："None"。
            - **atom_charge** (Tensor) - 原子电荷数。默认值："None"。
            - **atomic_number** (Tensor) - 原子序数。默认值："None"。

    .. py:method:: broadcast_multiplicity(multi_system)

        将信息广播到所选择的多系统中。

        参数：
            - **multi_system** (int) - 多系统中系统的数量。

    .. py:method:: build_atom_charge(template)

        把原子电荷数附到原子的索引中。

        参数：
            - **template** (Union[dict, str]) - 残基的模板。

    .. py:method:: build_atom_mass(template)

        把原子的质量附到原子的索引中。

        参数：
            - **template** (Union[dict, str]) - 残基的模板。

    .. py:method:: build_atom_type(template)

        把原子种类附到原子的索引中。

        参数：
            - **template** (Union[dict, str]) - 残基的模板。

    .. py:method:: build_atomic_number(template)

        把原子数附到原子的索引中。

        参数：
            - **template** (Union[dict, str]) - 残基的模板。

    .. py:method:: build_bond(template)

        把原子的边附到原子的索引中。

        参数：
            - **template** (Union[dict, str]) - 残基的模板。

    .. py:method:: set_name(name)

        设定残基的残基名。

        参数：
            - **name** (str) - 残基名称。

    .. py:method:: set_start_index(start_index)

        设定残基中第一个原子的开始索引。

        参数：
            - **start_index** (int) - 残基中第一个原子的开始索引。