sponge.colvar.TransformCV
==============================

.. py:class:: sponge.colvar.TransformCV(colvar: Colvar, function: Callable, periodic: bool = False, shape: Tuple[int] = None, unit: str = None, name: str = 'transform')

    使用特定函数 :math:`f(x)` 转换集合变量 :math:`s(R)` 的值。

    .. math::

        s' = f[s(R)]

    参数：
        - **colvar** (Colvar) - 集合变量(CVs) :math:`s(R)`。
        - **function** () - 变换函数 :math:`f(x)`。
        - **periodic** (bool) - 变换后的集合变量是否为周期性的。默认值：``False``。
        - **shape** (Tuple[int]) - 变换后的集合变量的shape。如果给出 ``None`` ，那么它将被分配为原始 `colvar` 的shape。默认值：``None``。
        - **unit** (str) -  集合变量的单位。默认值：``None``。注意：这不是包裹长度和能量的 `Units` 单元格。
        - **name** (str) -  集合变量的名称。默认值：'transform'。

    .. py:method:: set_pbc(use_pbc: bool)

        设置是否使用周期边界条件。

    .. py:method:: construct(coordinate: Tensor, pbc_box: Tensor = None)

        返回集合变量(CVs)经函数变换的值。

        参数：
            - **coordinate** (Tensor) - shape为 (B, A, D) 的张量。数据类型为float。colvar 在系统中的位置坐标。其中，B表示批量大小，即模拟中的步行者数量。A表示系统中的原子数。D表示仿真系统的维度。通常为3。
            - **pbc_box** (Tensor) - shape为 (B, D) 的张量。数据类型为float。PBC box的张量。默认值：``None``。

        返回：
            combine(Tensor): shape为 (B, S_1, S_2, ..., S_n) 的张量。数据类型为float。{S_i}表示集合变量的维度。