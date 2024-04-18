sponge.colvar.get_colvar
============================

.. py:function:: sponge.colvar.get_colvar(colvar_: Union[Colvar, List[Colvar], Tuple[Colvar]], axis: int = -1, use_pbc: bool = None, name: str = None)

    获取集合变量组。

    参数：
        - **colvar** (Union[Colvar, List[Colvar], Tuple[Colvar]]) - Colvar 或 colvars 数组。
        - **axis** (int) - 要连接的轴。默认值：-1。
        - **use_pbc** (bool) - 是否使用周期边界条件。默认值： ``None``。
        - **name** (str) - 名称。默认值： ``None``。

    返回：
        colvar(Colvar)：Colvar或ColvarGroup。