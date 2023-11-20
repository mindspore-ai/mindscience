sponge.colvar.Center
=========================

.. py:class:: Center(atoms: Union[AtomsBase, Tensor, ndarray, list], mass: Union[Tensor, ndarray, list] = None, batched: bool = False, keep_in_box: bool = False, keepdims: bool = None, axis: int = -2, name: str = 'atoms_center')

    特定原子的中心

    参数：
        - **atoms** (Union[AtomsBase, Tensor, ndarray, list]) - 特定原子或虚拟原子的shape (..., G, D) 。其中，G表示被平均的原子组的数目。D表示仿真系统的维度。通常为3。
        - **mass** (Union[Tensor, ndarray, list]) - 用于计算质心（COM）的原子质量数组。张量的shape为 (..., G) 或 (B, ..., G) ，数据类型是float。如果为空，则将计算坐标的几何中心。默认值：``None``。其中，B表示批量大小，即模拟中的步行者数量。
        - **batched** (bool) - index 和 mass 的第一维度是否为批大小。默认值：``False``。
        - **keep_in_box** (bool) - 是否在PBC box中替换坐标。默认值：``False``。
        - **keepdims** (bool) - 如果设置为 True，则缩小的轴将保留，以及中心的shape (..., 1, D) 。如果设置为 False，则中心的shape将为 (..., D) 。如果为 None，则其值将根据输入原子：如果秩大于 2，则为 False，否则为 True。默认值：``None``。
        - **axis** (int) - 计算位置平均值的轴。默认值：-2。
        - **name** (str) - Colvar的名称。默认值：'atoms_center'。
    
    支持的平台：
        ``Ascend`` ``GPU``

    .. py:method:: set_mass(mass: Tensor, batched: bool = False)

        设置原子块
    
    .. py:method:: construct(coordinate: Tensor, pbc_box: Tensor = None)
        
        计算特定原子的中心位置坐标

        参数：
            - **coordinate** (Tensor) - 张量的shape (B, A, D) 。数据类型为float。原子在系统中的位置坐标。其中，A代表系统中的原子数。
            - **pbc_box** (Tensor) - 张量的shape (B, D) 。数据类型为float。PBC box的张量。默认值：``None``。

        返回：
            中心位置(Tensor)：张量的shape (B, ..., D) 。数据类型为float。原子的中心坐标。