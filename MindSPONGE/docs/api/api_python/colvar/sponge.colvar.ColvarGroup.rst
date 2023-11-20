sponge.colvar.ColvarGroup
==============================

.. py:class:: ColvarGroup(colvar: Union[List[Colvar], Tuple[Colvar]], axis: int = -1, use_pbc: bool = None, name: str = 'colvar_group')

    将一组`Colvar`类连接成一个`Colvar`类

    参数：
        - **colvar** (list or tuple) - 要连接的Colvar数组。
        - **axis** (int) - 要连接的轴。注意: 这是指shape (B, S_1, S_2, ..., S_n) 的输出张量的轴。默认值: -1。
        - **use_pbc** (bool) - 是否使用周期边界条件。默认值: ``None``。
        - **name** (str) - 集合变量的名称。默认值: 'colvar_group'。

    支持的平台：
        ``Ascend`` ``GPU``

    .. py:method:: set_pbc(use_pbc: bool)

        设置是否使用周期性边界条件。

    .. py:method:: construct(coordinate: Tensor, pbc_box: Tensor = None)

        获取colvar组的位置坐标

        参数:
            - **coordinate** (Tensor) - 系统中原子位置坐标，shape为 (B, A, D) ，数据类型为float。
                                        其中B是批量大小，即模拟中的步行者数量。A是系统中的原子数。D是仿真系统的维度。通常为3。
            - **pbc_box** (Tensor) - PBC box，shape为 (B, D) ，数据类型为float。默认值： ``None``。

        返回：
            Tensor。 位置，shape为 (B, S_1, S_2, ..., S_n) 。数据类型为float。
