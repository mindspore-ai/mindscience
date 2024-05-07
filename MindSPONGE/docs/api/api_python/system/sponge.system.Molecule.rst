sponge.system.Molecule
==========================

.. py:class:: sponge.system.Molecule(atoms: Union[List[Union[str, int]], ndarray] = None, atom_name: Union[List[str], ndarray] = None, atom_type: Union[List[str], ndarray] = None, atom_mass: Union[Tensor, ndarray, List[float]] = None, atom_charge: Union[Tensor, ndarray, List[float]] = None, atomic_number: Union[Tensor, ndarray, List[float]] = None, bonds: Union[Tensor, ndarray, List[int]] = None, coordinate: Union[Tensor, ndarray, List[float]] = None, pbc_box: Union[Tensor, ndarray, List[float]] = None, template: Union[dict, str, List[Union[dict, str]]] = None, residue: Union[Residue, List[Residue]] = None, length_unit: str = None, **kwargs)

    分子体系层。

    参数：
        - **atoms** (Union[List[Union[str, int]], ndarray]) - 体系中的原子。数据可以是原子名称的字符串，也可以是原子序号的int值。默认值： ``None`` 。
        - **atom_name** (Union[List[str], ndarray]) - 原子名称字符串的array。默认值： ``None`` 。
        - **atom_type** (Union[List[str], ndarray]) - 原子种类字符串的array。默认值： ``None`` 。
        - **atom_mass** (Union[Tensor, ndarray, List[float]]) - 原子质量的array，类型为float，shape为 :math:`(B, A)` 其中B表示batch size， A表示原子数量。默认值： ``None`` 。
        - **atom_charge** (Union[Tensor, ndarray, List[float]]) - 原子电荷数的array，类型为float，shape为 :math:`(B, A)` 。默认值： ``None`` 。
        - **atomic_number** (Union[Tensor, ndarray, List[float]]) - 原子序数的array，类型为int，shape为 :math:`(B, A)` 。默认值： ``None`` 。
        - **bond** (Union[Tensor, ndarray, List[int]]) - 键连接的array，数据类型为int，shape为 :math:`(B, b, 2)` 其中b表示键数量。默认值： ``None`` 。
        - **coordinate** (Union[Tensor, ndarray, List[float]]) - 原子位置坐标 :math:`R` 的Tensor，shape为 :math:`(B, A, D)` 其中D表示模拟体系的维度，一般为3，数据类型为float。默认值： ``None`` 。
        - **pbc_box** (Union[Tensor, ndarray, List[float]]) - 周期性边界条件的box，shape为 :math:`(B, D)` 或者 :math:`(1, D)` 。默认值： ``None`` 。
        - **template** (Union[dict, str, List[Union[dict, str]]]) - 分子的模板。可以是一个MindSPONGE模板格式的字典，也可以是一个MindSPONGE模板文件的字符串。如果输入是一个字符串，该类会优先在MindSPONGE模板的构建路径下(`mindsponge.data.template` )搜索与输入同名的文件。默认值： ``None`` 。
        - **residue** (Union[Residue, List[Residue]]) - 残基或残基列表。如果 `template` 不是 ``None`` 的话，只有模板里的残基会被使用。默认值： ``None`` 。
        - **length_unit** (str) - 长度单位。如果为 ``None`` ，则使用全局长度单位。默认值： ``None`` 。
        - **kwargs** (dict) - 其他参数，用于扩展。

    输出：
        - 坐标，shape为 :math:`(B, A, D)` 的Tensor，其中B表示batch size， A表示原子数量，D表示模拟体系的维度，一般为3。数据类型为float。
        - 周期性边界条件盒子，shape为 :math:`(B, D)` 的Tensor，其中B表示batch size， D表示模拟体系的维度，一般为3。数据类型为float。


    .. py:method:: add_residue(residue: Residue, coordinate: Tensor = None)

        向当前分子系统增加残基。

        参数：
            - **residue** (class) - 向系统中增加的残基的 `Residue` 类。
            - **coordinate** (Tensor) - 输入残基的坐标。默认值： ``None`` 。

    .. py:method:: append(system)

        向当前分子系统添加系统。

        参数：
            - **system** (Molecule) - 添加进该分子系统的另一个分子系统。

    .. py:method:: build_angle()

        构建系统的角度。

    .. py:method:: build_atom_charge()

        构建原子电荷数。

    .. py:method:: build_atom_type()

        构建原子种类。

    .. py:method:: build_dihedrals()

        构建系统的二面角。

    .. py:method:: build_h_bonds()

        构建氢原子键。

    .. py:method:: build_improper()

        构建系统不适当的二面角。

    .. py:method:: build_space(coordinate: Tensor, pbc_box: Tensor = None)

        构建坐标系和周期性边界条件箱。

        参数：
            - **coordinate** (Tensor) - 系统的初始坐标。如果是 ``None`` ，系统会随机生成一个坐标作为它的初始坐标。
            - **pbc_box** (Tensor) - 系统的初始周期性边界条件箱。如果是 ``None`` ，则系统不会使用周期性边界系统。默认值： ``None`` 。

    .. py:method:: build_system()

        通过残基构建系统。

    .. py:method:: calc_colvar(colvar: Colvar) -> Tensor

        计算系统中特定的集体变量的值。

        参数：
            - **colvar** (Colvar) - 一般的集体变量 :math:`s(R)` 的基类。

        返回：
            Tensor，集体变量 :math:`s(R)` 的值。

    .. py:method:: calc_image(shift: float = 0) -> Tensor

        计算坐标图。

        参数：
            - **shift** (float) - 相对于箱子尺寸 :math:`\vec{L}` 的偏移比 :math:`c` 。默认值： ``0`` 。

        返回：
            Tensor，坐标图。

    .. py:method:: convert_length_from(unit)

        从指定的单位转换长度。

        参数：
            - **unit** (Union[str, Units, Length, float, int]) - 长度单位。

        返回：
            float，从指定单位转换所得长度。

    .. py:method:: convert_length_to(unit)

        把长度转换到指定的单位。

        参数：
            - **unit** (Union[str, Units, Length, float, int]) - 长度单位。

        返回：
            float，根据特定单位换算所得长度。

    .. py:method:: coordinate_in_pbc(shift: float = 0) -> Tensor

        获取在整个周期性边界条件箱中的坐标。

        参数：
            - **shift** (float) - 相对于箱子尺寸的偏移比。默认值： ``0`` 。

        返回：
            Tensor，周期性边界条件箱中的坐标。shape为 `(B, ..., D)` ，数据类型为float。

    .. py:method:: copy(shift: Tensor = None)

        返回一个复制当前 `Molecule` 参数的 `Molecule` 类。

        参数：
            - **shift** (Tensor) - 系统的移动距离。默认值： ``None`` 。

        返回：
            class，复制了当前 `Molecule` 类的参数的 `Molecule` 类。

    .. py:method:: fill_water(edge: float = None, gap: float = None, box: ndarray = None, pdb_out: str = None, template: str = None)

        Molecule类中给周期性边界条件箱加水的内部方法。

        参数：
            - **edge** (float) - 系统周围水的边长，默认值 ``None`` 。
            - **gap** (float) - 系统原子和水原子之间的最小间隔，默认值 ``None`` 。
            - **box** (Tensor) - 周期性边界条件箱，默认值 ``None`` 。
            - **pdb_out** (str) - 存放加水后的系统信息的pdb文件的名字，默认值 ``None`` 。
            - **template** (str) - 加的水分子的补充模板，默认值 ``None`` 。

        返回：
            Tensor，加水后的周期性边界条件箱。

    .. py:method:: get_atoms(atoms: Union[Tensor, Parameter, ndarray, str, list, tuple])

        从系统中获取原子。

        参数：
            - **atoms** (Union[Tensor, Parameter, ndarray, str, list, tuple]) - 原子列表。

        返回：
            class。原子或一些原子。

    .. py:method:: get_coordinate(atoms: AtomsBase = None)

        获取坐标的Tensor。

        参数：
            - **atoms** (class) - 特殊原子群的基类，在MindSPONGE中被用作 `atoms group module` 。默认值： ``None`` 。

        返回：
            Tensor，坐标。数据类型为float。

    .. py:method:: get_pbc_box()

        获取周期性边界条件箱。

        返回：
            Tensor。周期性边界条件箱。

    .. py:method:: get_volume()

        获得系统的容积。

        返回：
            Tensor。系统的容积。如果没有使用周期性边界条件箱，容积为None。

    .. py:method:: heavy_atom_mask()
        :property:

        重原子（非氢原子）的掩码。

        返回：
            Tensor，重原子的掩码。

    .. py:method:: length_unit()
        :property:

        长度单位。

        返回：
            str，长度单位。

    .. py:method:: move(shift: Tensor = None)

        移动系统的坐标。

        参数：
            - **shift** (Tensor) - 系统的移动距离。默认值： ``None`` 。

    .. py:method:: ndim()
        :property:

        原子坐标的维度数量。

        返回：
            int，原子坐标的维度的数量。

    .. py:method:: reduplicate(shift: Tensor)

        复制系统让其扩大到原来的两倍。

        参数：
            - **shift** (Tensor) - 从原始系统移动的距离。

    .. py:method:: repeat_box(lattices: list)

        根据周期性边界条件的box的格点重复系统。

        参数：
            - **lattices** (list) - 周期性边界条件箱的格点。

    .. py:method:: residue_bond(res_id: int)

        获得残基键的索引。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。残基键的索引。

    .. py:method:: residue_coordinate(res_id: int)

        获得残基坐标。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。系统中残基的坐标。

    .. py:method:: residue_head(res_id: int)

        获取残基的头索引。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。残基的头索引。

    .. py:method:: residue_index(res_id: int)

        获得残基的索引。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。残基在系统中的索引。

    .. py:method:: residue_tail(res_id: int)

        获得残基的尾索引。

        参数：
            - **res_id** (int) - 残基ID参数。

        返回：
            Tensor。残基的尾索引。

    .. py:method:: set_atom_charge(atom_charge: Tensor)

        设置原子电荷。

        参数：
            - **atom_charge** (Tensor) - 原子电荷。

    .. py:method:: set_bond_length(bond_length: Tensor)

        设置键长。

        参数：
            - **bond_length** (Tensor) - 设置系统的键长。

    .. py:method:: set_coordianate(coordinate: Tensor)

        设定坐标的值。

        参数：
            - **coordinate** (Tensor) - 用于设定系统坐标的坐标。

        返回：
            Tensor，系统的坐标。

    .. py:method:: set_length_unit(unit)

        设定系统的长度单位。

        参数：
            - **unit** (Union[str, Units, Length, float, int]) - 长度单位。

    .. py:method:: set_pbc_box(pbc_box: Tensor = None) -> Tensor

        设置周期性边界条件箱。

        参数：
            - **pbc_box** (Tensor) - 设置系统的周期性边界条件箱。如果是None，系统不会使用周期性边界条件箱。默认值： ``None`` 。

        返回：
            Tensor，系统的周期性边界条件箱。

    .. py:method:: set_pbc_grad(grad_box: bool)

        设置是否计算周期性边界条件箱的梯度。

        参数：
            - **grad_box** (bool) - 是否计算周期性边界条件箱的梯度。

        返回：
            bool，是否计算周期性边界条件箱的梯度。

    .. py:method:: shape()
        :property:

        原子坐标的shape。

        返回：
            Tensor，原子坐标的shape。

    .. py:method:: space_parameters()

        获取空间的参数(坐标和周期性边界条件箱)。

        返回：
            list。坐标和周期性边界条件箱。如果周期性边界条件箱未使用，则只返回坐标。

    .. py:method:: trainable_params(recurse=True)

        获取可训练参数。

        参数：
            - **recurse** (bool) - 如果为True，则产生此网络层和所有子网络层的参数。否则，只产生作为此网络层直接成员的参数。默认值： ``True`` 。

        返回：
            list，所有可训练参数的list。

    .. py:method:: update_coordinate(coordinate: Tensor)

        更新坐标的参数。

        参数：
            - **coordinate** (Tensor) - 用于更新系统坐标的坐标。

        返回：
            Tensor。更新后的系统坐标。

    .. py:method:: update_image(image: Tensor=None)

        更新坐标图。

        参数：
            - **image** (Tensor) - 用于更新系统坐标图的坐标图。默认值： ``None`` 。

        返回：
            bool，是否成功更新了系统坐标图。

    .. py:method:: update_pbc_box(pbc_box: Tensor)

        更新周期性边界条件箱。

        参数：
            - **pbc_box** (Tensor) - 用于更新系统周期性边界条件箱的周期性边界条件箱。

        返回：
            Tensor，更新后的周期性边界条件箱。