sponge.colvar.FunctionCV
============================

.. py:class:: sponge.colvar.FunctionCV(colvar: Union[Colvar, List[Colvar], Tuple[Colvar]], function: Callable, periodic: bool, shape: Tuple[int] = None, unit: str = None, use_pbc: bool = None, name: str = 'function')

    组合一组集合变量（CVs）的复合 Colvar :math:`{s_i(R)}` 使用自定义函数 :math:`f(s_1(R), s_2(R), ... s_i(R))`。

    .. math::

        S = f(s_1(R), s_2(R), ... s_i(R))

    参数：
        - **colvar** (Union[Colvar, List[Colvar], Tuple[Colvar]]) - 要组合的集合变量 :math:`{s_i(R)}`。
        - **function** (callable) - 自定义函数 :math:`f(s_1(R), s_2(R), ... s_i(R))`。
        - **periodic** (bool) - 自定义集合变量是否为周期性变量。
        - **shape** (tuple) - 自定义集合变量的形状。如果给出空并且所有 CVs 都在 `colvar` 中具有相同的shape，然后它将被分配shape。如果每个 CVs 的shape在 `colvar` 中不完全一样，必须设置 `shape`。默认值：``None``。
        - **unit** (str) - 集合变量的单位。默认值：``None``。注意：这不是包裹长度和能量的 `Units` 单元格。
        - **name** (str) - 集合变量的名称。默认值：'combine'。

    .. py:method:: set_pbc(use_pbc: bool)

        设置是否使用周期边界条件。

    .. py:method:: construct(coordinate: Tensor, pbc_box: Tensor = None)

        获取 colvar 组的位置坐标。

        参数：
            - **coordinate** (Tensor) - shape为 (B, A, D) 的张量。数据类型为float。colvar 在系统中的位置坐标。其中，B表示批量大小，即模拟中的步行者数量。A表示系统中的原子数。D表示仿真系统的维度。通常为3。
            - **pbc_box** (Tensor) - shape为 (B, D) 的张量。数据类型为float。PBC box的张量。默认值：``None``。

        返回：
            combine(Tensor): shape为 (B, S_1, S_2, ..., S_n) 的张量。数据类型为float。{S_i}表示集合变量的维度。