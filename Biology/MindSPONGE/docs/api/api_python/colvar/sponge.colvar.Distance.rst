sponge.colvar.Distance
===========================

.. py:class:: sponge.colvar.Distance(atoms: AtomsBase = None, atoms0: AtomsBase = None, atoms1: AtomsBase = None, vector: Vector = None, use_pbc: bool = None, batched: bool = False, keepdims: bool = None, axis: int = -2, name: str = 'distance')

    距离的集合变量。

    参数：
        - **atoms** (AtomsBase) - 用shape为 (..., 3, D) 的原子计算shape为 (...) 或 (..., 1) 的距离。不能与 `atoms0` 或 `atoms1` 一起使用。默认值： ``None``。其中，D表示仿真系统的维度。通常为3。
        - **atoms0** (AtomsBase) - shape为 (..., D) 的原子与shape为 (...) 或 (..., 1) 的距离的初始点。必须与 `atoms1` 一起使用，不能与 `atoms` 一起使用。默认值： ``None``。
        - **atoms1** (AtomsBase) - shape为 (..., D) 的原子的终点与shape为 (...) 或 (..., 1) 的距离。必须与 `atoms0` 一起使用，不能与 `atoms` 一起使用。默认值： ``None``。
        - **vector** (Vector) - shape为 (..., D) 的向量与shape为 (...) 或 (..., 1) 的距离。
        - **use_pbc** (bool) - 是否在周期边界条件下计算距离。默认值： ``None``。
        - **batched** (bool) - 判断以原子为单位的输入索引的第一维是否为批大小。默认值： ``False``。
        - **keepdims** (bool) - 如果为 True，则最后一根轴将保留，输出shape为 (..., 1) 。如果为 False，则距离的shape将为 (...) 。如果是 None，则其值将根据向量的秩：如果秩大于 1，则为False，否则为 True。默认值： ``None``。
        - **axis** (int) - 沿其取原子坐标的轴，其维度必须为 2。仅在使用 `atoms`、 `atoms0` 或 `atoms1` 初始化时有效。默认值：-2。
        - **name** (str) - Colvar的名称。默认值：'distance'。

    .. py:method:: get_unit(units: Units = None)

        集合变量的返回单位。