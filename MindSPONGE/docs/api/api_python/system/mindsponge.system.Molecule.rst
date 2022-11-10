mindsponge.system.Molecule
==========================

.. py:class:: mindsponge.system.Molecule(atoms=None, atom_name=None, atom_type=None, atom_mass=None, atom_charge=None, atomic_number=None, bond=None, coordinate=None, pbc_box=None, template=None, residue=None, length_unit=None)

    分子体系层。

    参数：
        - **atoms** (list) - 体系中的原子。默认值："None"。
        - **atom_name** (list) - 原子名称。默认值："None"。
        - **atom_type** (list) - 原子种类。默认值："None"。
        - **atom_mass** (Tensor) - 原子质量，shape为(B, A)。默认值："None"。
        - **atom_charge** (Tensor) - 原子电荷数，shape为(B, A)。默认值："None"。
        - **atomic_number** (Tensor) - 原子序数，shape为(B, A)。默认值："None"。
        - **bond** (Tensor) - 边的索引，shape为(B, b, 2)或者(1, b, 2)。默认值："None"。
        - **coordinate** (Tensor) - 原子位置坐标，shape为(B, A, D)或者(1, A, D)。默认值："None"。
        - **pbc_box** (Tensor) - 周期性边界条件的box，shape为(B, D)或者(1, D)。默认值："None"。
        - **template** (Union[dict, str]) - 残基的模板。默认值："None"。
        - **residue** (Union[dict, str]) - 残基系数。默认值："None"。
        - **length_unit** (str) - 位置坐标的长度单位。默认值："None"。

    符号：
        - **B** - Batch size。
        - **A** - 原子数量。
        - **b** - 边数量。
        - **D** - 模拟体系的维度，一般为3。

    .. py:method:: add_residue(residue, coordinate=None)

        增加残基。

        参数：
            - **residue** (Union[Residue, list]) - 残基参数。
            - **coordinate** (Tensor) - 原子的位置坐标，shape为(B, A, D)或者(1, A, D)。默认值："None"。

    .. py:method:: append(system)

        添加系统。

        参数：
            - **system** (Molecule) - 系统参数。

    .. py:method:: build_atom_charge()

        构建原子电荷数。

    .. py:method:: build_atom_type()

        构建原子种类。

    .. py:method:: build_space(coordinate, pbc_box=None)

        构建坐标系和周期性边界条件box。
    
        参数：
            - **coordinate** (Tensor) - 原子的位置坐标。
            - **pbc_box** (Tensor) - 周期性边界条件box。默认值："None"。

    .. py:method:: build_system()

        通过残基构建系统。

    .. py:method:: calc_image(shift=0.0)

        计算坐标图。

        参数：
            - **shift** (float) - 转换参数。默认值：0.0。

        返回：
            Tensor。坐标图。

    .. py:method:: coordinate_in_box(shift=0)

        获取整个周期性边界条件box中的坐标。

        参数：
            - **shift** (float) - 转换参数。默认值：0.0。

        返回：
            Tensor。整个周期性边界条件box中的坐标。

    .. py:method:: copy(shift=None)

        返回一个复制当前分子参数的分子。

        参数：
            - **shift** (Tensor) - 转换参数。默认值："None"。

    .. py:method:: get_coordinate()

        获取坐标的Tensor。

        返回：
            Tensor。坐标的Tensor。

    .. py:method:: get_pbc_box()

        获取周期性边界条件box。

        返回：
            Tensor。周期性边界条件box。

    .. py:method:: get_volume()

        获得系统的容积。

        返回：
            Tensor。系统的容积。

    .. py:method:: move(shift=None)

        移动系统的坐标。

        参数：
            - **shift** (Tensor) - 转换参数。默认值："None"。

    .. py:method:: reduplicate(shift)

        复制系统让其扩大到原来的两倍。

        参数：
            - **shift** (Tensor) - 转换参数。

    .. py:method:: repeat_box(lattices)

        根据周期性边界条件的box的格点重复系统。

        参数：
            - **lattices** (list) - 格点参数。

    .. py:method:: residue_bond(res_id)

        获得残基的边的索引。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。残基的边的索引。

    .. py:method:: residue_coordinate(res_id)

        获得残基坐标。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。残基的边的索引。

    .. py:method:: residue_head(res_id)

        获取残基的头索引。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。残基的头索引。

    .. py:method:: residue_index(res_id)

        获得残基索引。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。残基的索引。

    .. py:method:: residue_tail(res_id)

        获得残基的尾索引。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。残基的尾索引。

    .. py:method:: set_bond_length(bond_length)

        设置边的长度。

        参数：
            - **bond_length** (Tensor) - 边的长度。

    .. py:method:: set_coordianate(coordinate)

        设定坐标的值。

        参数：
            - **coordianate** (Tensor) - 原子的位置坐标。

    .. py:method:: set_length_unit(unit)

        设定系统的长度单位。

        参数：
            - **unit** (Units) - 长度单位。

    .. py:method:: set_pbc_box(pbc_box=None)

        设置周期性边界条件box。

        参数：
            - **pbc_box** (Tensor) - 周期性边界条件box。默认值："None"。

    .. py:method:: set_pbc_grad(grad_box)

        设置是否计算周期性边界条件box的梯度。

        参数：
            - **grad_box** (bool) - 是否计算周期性边界条件box的梯度。

    .. py:method:: space_parameters()

        获取空间的参数(坐标和周期性边界条件box)。

        返回：
            list。空间参数的list。

    .. py:method:: trainable_params(recurse=True)

        获取可训练参数。

        参数：
            - **recurse** (bool, 可选) - 递归参数。默认值："True"。

        返回：
            list。可训练参数list。

    .. py:method:: update_coordinate(coordinate, success=True)

        更新坐标的参数。

        参数：
            - **coordinate** (Tensor) - 原子的位置坐标。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。是否更新了坐标的参数。

    .. py:method:: update_image(image=None, success=True)

        更新坐标图。

        参数：
            - **image** (Tensor) - 图参数。默认值："None"。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。

    .. py:method:: update_pbc_box(pbc_box, success=True)

        更新周期性边界条件box。

        参数：
            - **pbc_box** (Tensor) - 周期性边界条件box，shape为(B, D)或者(1, D)。
            - **success** (bool, 可选) - 判断是否成功的参数。默认值："True"。

        返回：
            bool。是否更新了周期性边界条件box。