sponge.colvar.ColvarGroup
==============================

.. py:class:: sponge.colvar.ColvarGroup(colvar: Union[List[Colvar], Tuple[Colvar]], axis: int = -1, use_pbc: bool = None, name: str = 'colvar_group')

    将一组 `Colvar` 类连接成一个 `Colvar` 类。

    参数：
        - **colvar** (list or tuple) - 要连接的Colvar数组。
        - **axis** (int) - 要连接的轴。注意: 这是指shape (B, S_1, S_2, ..., S_n) 的输出张量的轴。默认值： -1。
        - **use_pbc** (bool) - 是否使用周期边界条件。默认值： ``None``。
        - **name** (str) - 集合变量的名称。默认值： 'colvar_group'。

    .. py:method:: set_pbc(use_pbc: bool)

        设置是否使用周期性边界条件。