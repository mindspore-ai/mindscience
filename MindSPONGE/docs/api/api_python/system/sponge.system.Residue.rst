sponge.system.Residue
=========================

.. py:class:: sponge.system.Residue(atom_name: Union[List[str], ndarray] = None, atom_type: Union[List[str], ndarray] = None, atom_mass: Union[Tensor, ndarray, List[float]] = None, atom_charge: Union[Tensor, ndarray, List[float]] = None, atomic_number: Union[Tensor, ndarray, List[float]] = None, bond: Union[Tensor, ndarray, List[int]] = None, head_atom: int = None, tail_atom: int = None, start_index: int = 0, name: str = 'MOL', template: Union[dict, str] = None)

    残基的基类。 `Residue` 神经元是 `Molecule` (system) 的组成部分。 `Residue` 不止可以代表单一的氨基酸残基，还可以代表分子系统中的一个小分子，例如一个水分子，一个无机盐离子等。这代表着 `Residue` 和PDB文件中的 "residue" 有着相似的概念。

    .. Note::
        `Residue` 只用来表示原子属性和键连接关系，不包含原子坐标。

    参数：
        - **atom_name** (Union[List[str], ndarray]) - 原子名称的array，数据类型为str。默认值："None"。
        - **atom_type** (Union[List[str], ndarray]) - 原子种类的array，数据类型为str。默认值："None"。
        - **atom_mass** (Union[Tensor, ndarray, List[float]]) - 原子质量的array，shape为 `(B, A)` ，数据类型为float。默认值："None"。
        - **atom_charge** (Union[Tensor, ndarray, List[float]]) - 原子电荷的array，shape为 `(B, A)` ，数据类型为float。默认值："None"。
        - **atomic_number** (Union[Tensor, ndarray, List[float]]) - 原子序数的array，shape为 `(B, A)` ，数据类型为float。默认值："None"。
        - **bond** (Union[Tensor, ndarray, List[int]]) - 键连接的array，shape为 `(B, b, 2)` ，数据类型为int。默认值："None"。
        - **head_atom** (int) - 与前一个残基相连接的头原子的索引。默认值："None"。
        - **tail_atom** (int) - 与下一个残基相连的尾原子的索引。默认值："None"。
        - **start_index** (int) - 残基中第一个原子的开始索引。默认值：0。
        - **name** (str) - 残基名称。默认值：'MOL'。
        - **template** (Union[dict, str]) - 残基的模板。默认值："None"。

    符号：
        - **B** - Batch size。
        - **A** - 原子总数。
        - **b** - 键总数。

    .. py:method:: name()

        获取残基的名称。

        返回：
            str，残基名称。

    .. py:method:: build_atom_mass(template)

        按照模板中原子名称对应的原子索引，获取模板对应索引的原子质量并加到残基对应原子上。

        参数：
            - **template** (dict) - 残基的模板。

    .. py:method:: build_atomic_number(template)

        按照模板中原子名称对应的原子索引，获取模板对应索引的原子数并加到残基对应原子上。

        参数：
            - **template** (dict) - 残基的模板。

    .. py:method:: build_atom_type(template)

        按照模板中原子名称对应的原子索引，获取模板对应索引的原子种类并加到残基对应原子上。

        参数：
            - **template** (dict) - 残基的模板。

    .. py:method:: build_atom_charge(template)

        按照模板中原子名称对应的原子索引，获取模板对应索引的原子电荷数并加到残基对应原子上。

        参数：
            - **template** (dict) - 残基的模板。

    .. py:method:: build_bond(template)

        按照模板中原子名称对应的原子索引，获取模板对应索引的原子的化学键并加到残基对应原子上。

        参数：
            - **template** (Union[dict, str]) - 残基的模板。

    .. py:method:: add_atom(atom_name=None, atom_type=None, atom_mass=None, atom_charge=None, atomic_number=None)

        把一个原子添加到残基中。

        参数：
            - **atom_name** (str) - 原子名称。默认值："None"。
            - **atom_type** (str) - 原子种类。默认值："None"。
            - **atom_mass** (float) - 原子质量。默认值："None"。
            - **atom_charge** (float) - 原子电荷数。默认值："None"。
            - **atomic_number** (str) - 原子序数。默认值："None"。

    .. py:method:: broadcast_multiplicity(multi_system)

        将信息广播到所选择的多系统中。

        参数：
            - **multi_system** (int) - 多系统中系统的数量。

    .. py:method:: set_name(name)

        设定残基的残基名。

        参数：
            - **name** (str) - 残基名称。

    .. py:method:: set_start_index(start_index)

        设定残基的开始索引。

        参数：
            - **start_index** (int) - 残基的开始索引。