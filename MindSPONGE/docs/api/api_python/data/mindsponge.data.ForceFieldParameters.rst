mindsponge.data.ForceFieldParameters
====================================

.. py:class:: mindsponge.data.ForceFieldParameters(atom_types, parameters, atom_names, atom_charges)

    获取原子和边的种类的参数。

    参数：
        - **atom_types** (str) - 力场中定义的原子种类。
        - **parameters** (dict) - 存储了所有力场常数的字典。
        - **atom_names** (str) - 氨基酸中原子序列名称。
        - **atom_charges** (ndarray) - 氨基酸中原子的电荷序列。

    .. py:method:: check_improper(bonds, core_id)

        检查是否有相同的不当二面角。

        参数：
            - **bonds** (ndarray) - 边。
            - **core_id** (ndarray) - 中心索引。

        返回：
            int。相同的不当二面角的中心索引。

    .. py:method:: combinations(bonds, bonds_for_angle, middle_id)

        获取三个原子的所有连接关系。

        参数：
            - **bonds** (ndarray) - 边。
            - **bonds_for_angle** (ndarray) - 构成角的边。
            - **middle_id** (ndarray) - 中间ID。

        返回：
            ndarray。角。

    .. py:method:: construct_hash(bonds)

        构建哈希图。

        参数：
            - **bonds** (ndarray) - 边。

        返回：
            dict。哈希图。

    .. py:method:: get_angle_params(angles, atom_type)

        获取力场角参数。

        参数：
            - **angles** (ndarray) - 角。
            - **atom_type** (ndarray) - 原子种类。

        返回：
            dict。参数。

    .. py:method:: get_bond_params(bonds, atom_type)

        获取力场中键的参数。

        参数：
            - **bonds** (ndarray) - 两个原子之间的键。
            - **atom_type** (ndarray) - 原子种类。

        返回：
            dict。参数。

    .. py:method:: get_dihedral_params(dihedrals_in, atom_types)

        获取力场中二面角的参数。

        参数：
            - **dihedrals_in** (ndarray) - 输入的二面角。
            - **atom_types** (ndarray) - 原子种类。

        返回：
            dict。参数。

    .. py:method:: get_dihedrals(angles, dihedral_middle_id)

        获取二面角的索引。

        参数：
            - **angles** (ndarray) - 角。
            - **dihedral_middle_id** (ndarray) - 二面角中间索引。

        返回：
            ndarray。二面角。

    .. py:method:: get_excludes(bonds, angles, dihedrals, improper)

        获取被排除的原子索引。

        参数：
            - **bonds** (ndarray) - 边。
            - **angles** (ndarray) - 角。
            - **dihedrals** (ndarray) - 二面角。
            - **improper** (ndarray) - 不当的信息。

        返回：
            ndarray。被排除的原子索引。

    .. py:method:: get_hbonds(bonds)

        获取氢键。

        参数：
            - **bonds** (ndarray) - 边。

        返回：
            ndarray。氢键。
            ndarray。非氢键。

    .. py:method:: get_improper(bonds, core_id)

        获取不正确的二面角索引。

        参数：
            - **bonds** (ndarray) - 边。
            - **core_id** (ndarray) - 核心索引。

        返回：
            ndarray。不正确的二面角。
            ndarray。新的ID。

    .. py:method:: get_improper_params(improper_in, atom_types, third_id)

        获取非正确二面角的预处理。

        参数：
            - **improper_in** (ndarray) - 输入的不正确二面角。
            - **atom_types** (ndarray) - 原子种类。
            - **third_id** (ndarray) - 第三ID。

        返回：
            dict。参数。

    .. py:method:: get_pair_index(dihedrals, angles, bonds)

        获取非键原子对的索引。

        参数：
            - **dihedrals** (ndarray) - 二面角。
            - **angles** (ndarray) - 角。
            - **bonds** (ndarray) - 键。

        返回：
            ndarray。非键原子对的索引。

    .. py:method:: get_pair_params(pair_index, epsilon, sigma)

        获取所有成对参数。

        参数：
            - **pair_index** (ndarray) - 成对索引。
            - **epsilon** (ndarray) - 参数epsilon。
            - **sigma** (ndarray) - 参数sigma。

        返回：
            dict。成对参数。

    .. py:method:: get_pairwise_c6(e0, e1, r0, r1)

        在VDW势中计算B系数。

        参数：
            - **e0** (ndarray) - 系数1。
            - **e1** (ndarray) - 系数2。
            - **r0** (ndarray) - 系数3。
            - **r1** (ndarray) - 系数4。

        返回：
            ndarray。在VDW势中的B系数。

    .. py:method:: get_vdw_params(atom_type)

        ['H', 'HO', 'HS', 'HC', 'H1', 'H2', 'H3', 'HP', 'HA', 'H4', 'H5', 'HZ', 'O', 'O2', 'OH', 'OS', 'OP', 'C*', 'CI', 'C5', 'C4', 'CT', 'CX', 'C', 'N', 'N3', 'S', 'SH', 'P', 'MG', 'C0', 'F', 'Cl', 'Br', 'I', '2C', '3C', 'C8', 'CO']

        参数：
            - **atom_type** (ndarray) - 原子种类。

        返回：
            dict。参数。

    .. py:method:: trans_dangles(dangles, middle_id)

        构建二面角。

        参数：
            - **dangles** (ndarray) - 二面角。
            - **middle_id** (ndarray) - 中间索引。

        返回：
            ndarray。二面角。