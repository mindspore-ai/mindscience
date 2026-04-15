sponge.colvar.TransformCV
==============================

.. py:class:: sponge.colvar.TransformCV(colvar: Colvar, function: Callable, periodic: bool = False, shape: Tuple[int] = None, unit: str = None, name: str = 'transform')

    使用特定函数 :math:`f(x)` 转换集合变量 :math:`s(R)` 的值。

    .. math::

        s' = f[s(R)]

    参数：
        - **colvar** (Colvar) - 集合变量(CVs) :math:`s(R)`。
        - **function** (Callable) - 变换函数 :math:`f(x)`。
        - **periodic** (bool) - 变换后的集合变量是否为周期性的。默认值： ``False``。
        - **shape** (Tuple[int]) - 变换后的集合变量的shape。如果给出 ``None`` ，那么它将被分配为原始 `colvar` 的shape。默认值： ``None``。
        - **unit** (str) -  集合变量的单位。默认值： ``None``。注意：这不是包裹长度和能量的 `Units` 单元格。
        - **name** (str) -  集合变量的名称。默认值：'transform'。

    .. py:method:: set_pbc(use_pbc: bool)

        设置是否使用周期边界条件。